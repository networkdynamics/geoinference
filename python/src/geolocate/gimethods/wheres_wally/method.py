##
#  Copyright (c) 2015, David Jurgens
#
#  All rights reserved. See LICENSE file for details
##

import collections
import os.path
import logging

from sklearn import naive_bayes # import GaussianNB
from sklearn import svm
from sklearn import preprocessing
import numpy
import scipy.sparse
import gzip
import time


from collections import defaultdict

from geolocate import GIMethod, GIModel

from geolocate.geocoder import Geocoder


LOGGER = logging.getLogger(os.path.basename(__file__))

class Wheres_Wally_Model(GIModel):

    def __init__(self, user_id_to_location):
        self.user_id_to_location = user_id_to_location

    def infer_post_location(self, post):
        if not "user" in post:
            return None
        user = post["user"]

        if not "id" in user:
            return None

        # If we know this user's location, report their home location
        user_id = user['id']
        if user_id in self.user_id_to_location:
            return self.user_id_to_location[user_id]
        else:
            return None


    def infer_posts_by_user(self, posts):
        if len(posts) == 0:
            return None
        
        # Each post is assumed originate from the user's home location, so just
        # infer the first post's location
        home_location = self.infer_post_location(posts[0])
        if home_location is None:
            return None

        # Then fill the array of post locations with the home location
        locations = []
        for i in range(0, len(posts)):
            locations.append(home_location)

        return locations


class Wheres_Wally(GIMethod):
    def __init__(self):
        # Location is represented as a lat/lon geopy Point
        self.user_id_to_location = {}
        self.geocoder = None;
        self.unique_locations = set()
        self.id_to_location = {}

        # Mappings from feature names to their corresponding indices in a
        # feature vector
        self.pop_bin_feature_indices = {}
        self.reciprocal_feature_indices = {}
        self.triad_feature_indices = {}
        self.total_num_features = 0
        
        # The SVM classifier and feature vector scaler
        self.location_classifier = None
        self.location_vector_scaler = None



    def train_model(self, settings, dataset, model_dir):

        # Initialize the geocoder, which we'll use to resolve location strings.
        # We use the default name-to-location mapping unless the user has
        # specified otherwise.
        if 'location_source' in settings:
            self.geocoder = Geocoder(dataset=settings['location_source'])
        else:
            self.geocoder = Geocoder()


        # NOTE: The original paper used the directional friends/followers
        # network.  However, the paper was tested on a much smaller network
        # (9.8M edges), which doesn't scale when including the full network.  We
        # opt for using the bi-directional networks as these (1) provide a
        # stronger signal of social relationships and (2) significantly reduce
        # the memory requirement.
        LOGGER.debug('Loading mention network')        
        mention_network = dataset.bi_mention_network()

        # This dict will contain a mapping from user ID to an associated home
        # location, which is derived either from the location field (as in the
        # original paper), from GPS-tagged tweets, or from both
        user_to_home_loc = {}
        
        # For each of the users that we have in the network, see if we can
        # associate that user with a home location.
        all_users = set(mention_network.nodes_iter())
        
        LOGGER.debug('Calculating users with recognizable home location')
        num_users_processed = 0

        # Keep track of how many times each location occurred.  We'll filter
        # this down to only the most common locations
        location_counts = collections.Counter() 

        for user_id, home_loc in dataset.user_home_location_iter():
            
            if not user_id in all_users:
                continue
            
            # home_loc is a (lat,lon) tuple.  While this is accurate, we want to
            # coarsen the location data to decrease sparsity (i.e., more people
            # located in the same city location, despite slightly different
            # underlying lat/lon values).  Here, use the Geocoder to map the
            # lat/lon to a name and then back to a canonical lat/lon for that
            # name           
            canonical_lat_lon = self.geocoder.canonicalize(home_loc[0], home_loc[1])

            location_counts[canonical_lat_lon] += 1

            user_to_home_loc[user_id] = canonical_lat_lon
            num_users_processed += 1
            if num_users_processed % 500000 == 0:
                LOGGER.debug('Processed %s of the %s users, associated %s a known location (%s)'
                             % (num_users_processed, len(all_users), len(user_to_home_loc),
                                len(user_to_home_loc) / float(num_users_processed)))

        # Iterate through the locations pruning out those that do not occur more
        # than some threshold number of times
        num_locs_removed = 0
        for lat_lon, count in location_counts.iteritems():
            if count >= 20:
                self.unique_locations.add(lat_lon)
            else:
                num_locs_removed += 1
        LOGGER.debug('Saw %d locations, %d with at least 5 users, %d to be pruned'
                     % (len(location_counts), len(self.unique_locations), num_locs_removed))


        # Remove the home locations of users whose locations aren't in the
        # pruned list of minimum-frequency locations
        num_user_home_locs_removed = 0
        for user_id, loc in user_to_home_loc.items():
            if not loc in self.unique_locations:
                del user_to_home_loc[user_id]
                num_user_home_locs_removed += 1
        LOGGER.debug('After pruning removed home locations of %d users, %d still have homes'
                     % (num_user_home_locs_removed, len(user_to_home_loc)))
                

        # Create a bi-directional mapping from locations to unique
        # numeric identifiers.  This mapping will be used when
        # representing locations in the classifier feature space and
        # when converting classifier output to specific locations
        location_to_id = {}
        for loc in self.unique_locations:
            id_ = len(location_to_id)
            location_to_id[loc] = id_
            self.id_to_location[id_] = loc

        # Associate each location with its set of features
        n = len(self.unique_locations)

        # Each location has 7 features associated with it for classifying a
        # user's location.  The seven features per location are arranged next to
        # each other in the feature space.
        feature_offset = 0
        for loc in self.unique_locations:
            # Feat1: it's population bin (size approx.)
            self.pop_bin_feature_indices[loc] = feature_offset
            # Feat2: the number of reciprocal friends
            self.reciprocal_feature_indices[loc] = feature_offset + 1
            # Feat3-7: the bins indicating how many friends were in reciprocal
            # triads in that city
            for bin_num in range(0, 5):
                feat = "%s,%s:%s" % (loc[0], loc[1], bin_num)
                self.triad_feature_indices[feat] = feature_offset + bin_num + 2
            # Increment the feature offset so the next city's features don't
            # collide with this city's indices 
            feature_offset += 7
        
        # Set the total number of features seen 
        self.total_num_features = feature_offset
        LOGGER.debug('Saw %d unique locations, %d total featurs' 
                     % (len(self.unique_locations), feature_offset))

        LOGGER.debug('Associated %s of the %s users with a known location (%s unique)'
                     % (len(user_to_home_loc), len(all_users), len(self.unique_locations)))

        # The list of locations for each corresponding user in X
        B = []
        
        # Train the classifier based on users with known home locations
        LOGGER.debug("Generating feature vectors for training")
        X = scipy.sparse.lil_matrix((len(user_to_home_loc), 
                                     self.total_num_features), dtype=numpy.float64)
	print X
        row = 0
        total_nz = 0
        for user_id, location in user_to_home_loc.iteritems():

            # Skip users whose locations were omitted due to frequency filtering
            # or who have home locations but are not in the mention network
            #if not location in self.unique_locations or not user_id in all_users:
            #    continue

            # Fill the row in the matrix corresponding to this user's features
            nz = self.fill_user_vector(user_id, mention_network,
                                       user_to_home_loc, X, row)
            total_nz += nz
            
            # Get the index of this user's location
            location_id = location_to_id[location]
            B.append(location_id)
            row += 1
        X = X.tocsr()
        #X = X.toarray()

        LOGGER.debug("Generated training data for %d users, %d nz features, %f on average"
                     % (row, total_nz, float(total_nz) / row))
        

        # Convert the location list into a numpy array for use with scikit
        Y = numpy.asarray(B)

        if len(X.nonzero()[0]) == 0:
            LOGGER.warning("Too little training data seen and no user had non-zero feature "+
                           "values.  Cowardly aborting classification")
        else:
            # Use SVM classifier with a linear kernel.
            #
            # NOTE NOTE NOTE NOTE
            #
            # The original paper uses an RBF kernel with their SVM.  However,
            # this proved impossibly slow during testing, so a linear kernel was
            # used instead.  
            #
            # NOTE NOTE NOTE NOTE
            #
            # slow: self.location_classifier = svm.SVC(kernel='rbf')
            #self.location_classifier = svm.LinearSVC(dual=False)
            #self.location_classifier = svm.NuSVC(kernel='rbf', verbose=True, max_iter=1000)
            #self.location_classifier = naive_bayes.BernoulliNB()
            self.location_classifier = svm.LinearSVC(dual=False, loss='l2', penalty="l2",
                                                     tol=1e-2)

            # Note: we expect the vector representations to be sparse, so avoid mean
            # scaling since it would create dense vectors, which would blow up the
            # memory consumption of the model
            self.location_vector_scaler = preprocessing.StandardScaler(with_mean=False)
            
            # Learn the scaling parameters and then rescale the input            
            LOGGER.debug("Scaling feature vectors for training")
            X_scaled = self.location_vector_scaler.fit_transform(X.astype(numpy.float64))

            LOGGER.debug("Training classifier")
            self.location_classifier.fit(X_scaled, Y)
            LOGGER.debug("Finished training classifier")

            # Assign all the users some location, if we can figure it out
            users_assigned = 0
            users_seen = 0
            for user_id in all_users:
                users_seen += 1
                # If we know where to place this user, assign it to their home location
                if user_id in user_to_home_loc:
                    self.user_id_to_location[user_id] = user_to_home_loc[user_id]
                # Otherwise try to infer the location
                else:
                    location = self.infer_location(user_id, mention_network,
                                                   user_to_home_loc)
                    if not location is None:
                        self.user_id_to_location[user_id] = location
                        users_assigned += 1

                if users_seen % 100000 == 0:
                    LOGGER.debug((("Saw %d/%d users, knew location of %d, " +
                                   "inferred the location of %d (total: %d)")
                                  % (users_seen, len(all_users),
                                     len(self.user_id_to_location) - users_assigned,
                                     users_assigned,
                                     len(self.user_id_to_location))))

        LOGGER.debug((("Ultimately saw %d/%d users, knew location of %d, " +
                       "inferred the location of %d (total: %d)")
                      % (users_seen, len(all_users),
                         len(self.user_id_to_location) - users_assigned,
                         users_assigned,
                         len(self.user_id_to_location))))
                        

        # Short circuit early if the caller has specified that the model is not
        # to be saved into a directory
        if model_dir is None:
            return Wheres_Wally_Model(self.user_id_to_location)

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)         

        # Write the .tsv for human debugability too
        fh = gzip.open(os.path.join(model_dir, 'user-to-lat-lon.tsv.gz'), 'w')
        for user_id, loc in self.user_id_to_location.iteritems():
            fh.write("%s\t%s\t%s\n" % (user_id, loc[0], loc[1]));
        fh.close()

        return Wheres_Wally_Model(self.user_id_to_location)
               

    def infer_location(self, user_id, mention_network, user_to_home_loc):
        """
        Infers and returns the location of the provided users based on their
        features in the network
        """
        
        # Ensure that the model has been trained; otherwise, report an
        # empty classification
        if self.location_vector_scaler is None or self.location_classifier is None:
            return None

        # Convert the user's network-based features into a numeric vector
        X = scipy.sparse.lil_matrix((1, self.total_num_features), dtype=numpy.float64)
        self.fill_user_vector(user_id, mention_network, user_to_home_loc, X, 0)
        X = X.tocsr()                

        # Rescale the vector according to the training data's scaling
        user_vector_scaled = self.location_vector_scaler.transform(X)
        
        # Classify the results
        location_id = self.location_classifier.predict(user_vector_scaled)[0]
                
        # Convert the index into a location
        return self.id_to_location[location_id]
        

    def fill_user_vector(self, user_id, mention_network, user_to_home_loc,
                         csr_matrix, row_to_fill):
        """         
        Creates a vector for the user and fills their data into the
        specified row in the provided matrix
        """
        feat_dict = self.create_user_vector(user_id, mention_network, 
                                            user_to_home_loc)
        nz = 0
        for col, val in feat_dict.iteritems():
            csr_matrix[row_to_fill, col] = val
            nz += 1
        return nz


    def create_user_vector(self, user_id, mention_network, user_to_home_loc):
        """
        Creates a vector to use with SciPy that represents this user's features
        """

        # The binned location features look at all the locations of this user's
        # neighbors and then provide a weight for each location according to how
        # many of the user's friends are in that location multiplied by how
        # large the city is, which is represented as one of five bins

        location_to_friends = defaultdict(list)
        location_to_followers = defaultdict(list)
        num_friends = mention_network.degree(user_id)

        # Record which friend appear in each city
        for neighbor_id in mention_network.neighbors_iter(user_id):
            if neighbor_id in user_to_home_loc:
                location_name = user_to_home_loc[neighbor_id]
                location_to_friends[location_name].append(neighbor_id)
		location_to_followers[location_name].append(neighbor_id)


        # Since the vector is expected to be very sparse, create it as a dict
        # for the indices with non-zero feature values.
        classifier_input_vector = {}
        num_non_zero_features = 0

        # Each city/location generates 7 unique features in the best performing
        # system
        for city, followers_in_city in location_to_followers.iteritems():
            n = len(followers_in_city)

            # Feature 1: the city's bin multiplied by the number of users in the
            # city
            city_bin = self.get_city_bin(n)
            pop_bin_feature_index = self.pop_bin_feature_indices[city]
            classifier_input_vector[pop_bin_feature_index] = city_bin

        for city, friends_in_city in location_to_friends.iteritems():
            n = len(friends_in_city)

            # Feature 2: the percentage of friends with reciprocal edges at that
            # location
            num_reciprocal_friends = 0
            for n1 in friends_in_city:
                if mention_network.has_edge(n1, user_id):
                    num_reciprocal_friends += 1
                    num_non_zero_features += 1
            reciprocal_feature_index = self.reciprocal_feature_indices[city]
            classifier_input_vector[reciprocal_feature_index] = num_reciprocal_friends / n
            if num_reciprocal_friends > 0:
                num_non_zero_features += 1
                    
            # Features 3-7: the number of triads in the city
            triad_counter = collections.Counter()
            for n1 in friends_in_city:
                num_triads = 0
                for n2 in friends_in_city:
                    if mention_network.has_edge(n1, n2):
                        num_triads += 1

                # Decide which bin this user is in
                triad_counter[self.get_triad_bin(num_triads)] += 1

            for bin_num, count in triad_counter.iteritems():
                feat = "%s,%s:%s" % (city[0], city[1], bin_num)
                triad_bin_feature_index = self.triad_feature_indices[feat]
                classifier_input_vector[triad_bin_feature_index] = count / num_friends
                if count > 0:
                    num_non_zero_features += 1

        return classifier_input_vector
                

    def get_triad_bin(self, num_triads):
        """
        Returns which bin this count of the number of triads should be in
        """
        # Bins in the paper [0,5,10,20,40]
        if num_triads < 5:
            return 0
        elif num_triads < 10:
            return 1
        elif num_triads < 20:
            return 2
        elif num_triads < 40:
            return 3
        else:
            return 4

    def get_city_bin(self, city_size):
        """
        Returns which bin this count of the number of triads should be in
        """
        # Bins in the paper [1,2,4,12,57054] 
        if city_size <= 1:
            return 0
        elif city_size <= 2:
            return 1
        elif city_size <= 4:
            return 2
        elif city_size <= 12:
            return 3
        # This sould be 57054, but we use any value larger than 12 to
        # avoid the edge case where a city has more than 57k users
        else: 
            return 4

    def load_model(self, model_dir, settings):
        """
        Reads in the Where's Wally model from a gzipped .tsv
        """      

        user_id_to_location = {}
        model_file = gzip.open(os.path.join(model_dir, "user-to-lat-lon.tsv.gz"), 'r')
        for line in model_file:
            cols = line.split("\t")
            user_id = cols[0]
            lat = float(cols[1])
            lon = float(cols[2])
            user_id_to_location[user_id] = (lat, lon)

        model_file.close()
        return Wheres_Wally_Model(user_id_to_location)
