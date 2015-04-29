##
#  Copyright (c) 2015, Tyler Finethy
#
#  All rights reserved. See LICENSE file for details
##

"""
A twitter geolocation inference method based on, "Multiple Location Profiling for
Users and Relationships from Social Network and Content" by Rui Li, Shengjie Wang,
and Kevin Chen-Chuan Chang.

Author: Tyler Finethy
Date Created: August 2014
"""

#from geopy.point import Point
#from twokenize import tokenizeRawTweetText as tokenizer
#from geopy.distance import vincenty, great_circle
import numpy as np
import math
import logging
import os
import random
from geolocate.geocoder import Geocoder
from scipy.optimize import curve_fit
from haversine import haversine
from geolocate import GIMethod, GIModel
from collections import Counter, defaultdict
import gzip
try:
    import cPickle as pickle
except:
    import pickle


logger = logging.getLogger(__name__)


class MultiLocationMethod(GIMethod):
    """
    MultiLocationMethod extends GIMethod and is the interface for training and loading models.
    """
    train_called = False
    settings = None
    dataset = None
    model_dir = None
    load_called = False

    @staticmethod
    def clear():
        MultiLocationMethod.train_called = False
        MultiLocationMethod.settings = None
        MultiLocationMethod.model_dir = None
        MultiLocationMethod.load_called = False

    def __init__(self):
        pass

    def train_model(self,settings,dataset,model_dir=None):
        """
        Creates a Multi_Location object classifier and saves the object as a pickled file
        """
        MultiLocationMethod.train_called = True
        MultiLocationMethod.settings = settings
        MultiLocationMethod.dataset = dataset

        logger.debug("Starting MultiLocationMethod")

        # Returning a model that is just the modified mention network with the
        # added location of inferred users as previously stated, this may have
        # to be altered if we're attaching nodes to the network but for now --
        # optimization/space reasons -- we'll just store this.
        augmented_mention_network = MultiLocation(settings).return_network()
        user_id_to_location = {}
        for user_id in augmented_mention_network.nodes_iter():
            try:
                location = augmented_mention_network.node_data(user_id)
                if not location is None:
                    user_id_to_location[user_id] = location
            except KeyError:
                pass            
        model = MultiLocationModel(user_id_to_location)
        logger.debug("Finished MultiLocationMethod! Model has been created.")

        if model_dir:
            logger.debug("Storing model in %s" % model_dir)
            MultiLocationMethod.model_dir = model_dir
            filename = os.path.join(model_dir, 'user-to-lat-lon.tsv.gz')

            fh = gzip.open(filename, 'w')
            for user_id, loc in user_id_to_location.iteritems():
                fh.write("%s\t%s\t%s\n" % (user_id, loc[0], loc[1]))
            fh.close()

        return model

    def load_model(self, model_dir, settings=None):
        """
        Loads a MultiLocationModel object classifier from a pickled file.
        """
        self.load_called = True
        self.model_dir = model_dir
        user_id_to_location = {}
	count = 0
	fh = gzip.open(os.path.join(model_dir, 'user-to-lat-lon.tsv.gz'), 'r')
        for line in fh:
            cols = line.split("\t")
	    try:
		long(cols[0])
	        user_id_to_location[cols[0]] = (float(cols[1]), float(cols[2]))
	    except:
		count += 1
        fh.close()
        if settings:
            self.settings = settings
	print '%d invalid tweet ids in multi-location model' % count
        return MultiLocationModel(user_id_to_location)


class MultiLocationModel(GIModel):
    """
    MultiLocationModel extends GIModel and is the classifier for geo-inferring tweets.
    """
    num_posts_inferred = 0
    num_users_inferred = 0

    @staticmethod
    def clear():
        MultiLocationModel.num_posts_inferred = 0
        MultiLocationModel.num_users_inferred = 0


    def __init__(self, user_id_to_location):
        """
        Initializes the class based on the model/classifier created by MultiLocationMethod
        """
        self.user_id_to_location = user_id_to_location
        return

    def infer_post_location(self, post):
        """
        Returns the computed latitude/longitude given a tweet or post
        """
        MultiLocationModel.num_posts_inferred += 1
        user_id = post['user']['id_str']
        if user_id in self.user_id_to_location:
            return self.user_id_to_location[user_id]
        else:
            return None

    def infer_posts_by_user(self, posts):
        """
        Returns a list of locations given multiple tweets from a single user
        """
        MultiLocationModel.num_users_inferred += 1
        return [self.infer_post_location(post) for post in posts]



class MultiLocation(object):
    """
    MultiLocation is the implemented method from Multiple Location Profiling for Users and Relationships,
    from Social Network and Content by Rui Li, Shengjie Wang and Kevin Chen-Chuan Chang.
    """

    def __init__(self, settings):
        """
        Initializing class variables.
        """
        # the mention network that will store inferred locations in node_data
        self.mention_network = MultiLocationMethod.dataset.bi_mention_network()
        self.nodes = set(self.mention_network.nodes())

        #self.u_n, self.u_star are the sets of users with unknown and known locations respectively.
        self.u_n = set()
        self.u_star = set()

        #the set of all known venues
        self.venues = set()

        #list of all locations, and the co-occurences with a user.
        self.psi = Counter()

        #alpha and beta are the coefficients for eq.1 as per the paper
        self.alpha = -0.55
        self.beta = 0.0045

        #K is the total number of tweeting relationships
        self.K = 0

        #N_squared is the total number of user pairs
        self.N_squared = 0

        #S is the number of following relationships
        self.S = 0

        #geocoder is a forward/reverse geocoder for location -> lat/long and lat/lon -> location.
        if 'location_source' in settings:
            self.geocoder = Geocoder(dataset=settings['location_source'])
        else:
            self.geocoder = Geocoder()


        #F_r is the random following model Bernoulli distribution parameter
        self.F_r = None

        #T_r is the random tweeting model Bernoulli distribution parameter
        self.T_r = Counter()

        #mu and nu are the model selectors according to a bernoulli distribution
        self.mu = defaultdict(bool)
        self.nu = defaultdict(bool)

        #the multi-location list generated by the MLP
        self.user_multi_locations = defaultdict(list)

        #runs the model, populates all the variables and generates user_multi_locations
        self.run_model()

    def store_location_data(self):
        """
        Sets the node_data field with the relevant gold-standard location data from
        the bidirectional dataset.
        """
        num_users_seen = 0
        for user_id, loc in MultiLocationMethod.dataset.user_home_location_iter():
            if loc[0] == 0 and loc[1] == 0:
                continue
            try:
                self.mention_network.set_node_data(user_id, loc)
                self.u_star.add(user_id)
                num_users_seen += 1
                if num_users_seen % 100000 == 0:
                    logger.debug('Multilocation saw %d users' % num_users_seen)
            except KeyError:
                pass

    def find_locations(self):
        users_seen = 1
        for possible_posts in MultiLocationMethod.dataset.user_iter():
            users_seen += 1
            if users_seen % 1000000 == 0:
                logger.debug("Seen %d users" %users_seen)

            user_id = possible_posts['user_id']
            posts = possible_posts['posts']
            if len(posts) > 600: posts = posts[-600:]
            for post in posts:

                #twokenizer may be too computationally expensive here...
                #text = tokenizer(post['text'])
                text = post['text'].split()
                lc_text = []
                is_upper = []
                for s in text:
                    isup = s[0].isupper()
                    is_upper.append(isup)
                    if isup:
                        lc_text.append(s.lower())
                    else:
                        lc_text.append(s)

                i = 0
                n = len(text)
                while True:
                    if i >= n:
                        break

                    if not is_upper[i]:
                        i += 1
                        continue

                    is_up1 = i + 1 < n and is_upper[i+1]
                    first_two_with_space = None
                    first_two_with_tab = None

                    if i + 2 < n and is_upper[i+2] and is_up1:
                        w1 = lc_text[i]
                        w2 = lc_text[i+1]
                        w3 = lc_text[i+2]

                        first_two_with_space = w1 + " " + w2
                        s2 = first_two_with_space + " " + w3
                        location = self.geocoder.geocode(s2)
                        if not location is None:
                            self.record_user_location(s2, location, user_id)
                            i += 3
                            continue

                        s3 = first_two_with_space + "\t" + w3
                        location = self.geocoder.geocode(s3)
                        if not location is None:
                            self.record_user_location(s3, location, user_id)
                            i += 3
                            continue

                        first_two_with_tab = w1 + "\t" + w2
                        s4 = first_two_with_tab + "\t"  + w3
                        location = self.geocoder.geocode(s4)
                        if not location is None:
                            self.record_user_location(s4, location, user_id)
                            i += 3
                            continue

                        s5 = first_two_with_tab + " " + w3
                        location = self.geocoder.geocode(s5)
                        if not location is None:
                            self.record_user_location(s5, location, user_id)
                            i += 3
                            continue

                    elif i + 1 < n and is_up1:
                        w1 = lc_text[i]
                        w2 = lc_text[i+1]

                        if first_two_with_tab is None:
                            first_two_with_tab = w1 + "\t" + w2

                        location = self.geocoder.geocode(first_two_with_tab)
                        if not location is None:
                            self.record_user_location(first_two_with_tab, location, user_id)
                            i += 2
                            continue

                        if first_two_with_space is None:
                            first_two_with_space = w1 + " " + w2
                        location = self.geocoder.geocode(first_two_with_space)
                        if not location is None:
                            self.record_user_location(first_two_with_space, location, user_id)
                            i += 2
                            continue

                    else:
                        w1 = lc_text[i]
                        location = self.geocoder.geocode(w1)
                        if not location is None:
                            self.record_user_location(w1, location, user_id)

                    i += 1

    def record_user_location(self, location_name, location, user_id):
        try:
            self.mention_network.add_edge(user_id,location_name)
            self.mention_network.set_node_data(location_name,location)
        except:
            return
        self.venues.add(location_name)
        self.psi[location] += 1
        self.T_r[user_id] += 1
        self.K += 1
        return


    def compute_coefficients(self):
        """
        Computes the coefficients for equation (1) form the paper,
        P(f<i,j>|alpha,beta,x_i,y_i) = beta*distance(x_i,y_i)^alpha
        """

        def func_to_fit(x, a, b):
                return b * x ** a

        mentions_per_distance = Counter()
        following_relationship = Counter()

        # our networks are too large to generate these coefficients on each call...
        # this is about the same number of combinations as shown in the paper...
        n = 10000000
        #random_sample = random.sample(list(self.u_star),n)
        random_sample = list(self.u_star)
        number_of_users = len(self.u_star)

        # processed_combinations = 0
        # start_time = time.time()
        #for node_u, node_v in combinations(random_sample,2):
        for i in range(0,n):
            node_u, node_v = (random_sample[random.randint(0,number_of_users-1)],random_sample[random.randint(0,number_of_users-1)])
            if node_u == node_v: continue
            # if processed_combinations % 1000000 == 0:
            #     logger.debug("Took %f to process %d combinations..." % ((time.time() - start_time), processed_combinations))
            # processed_combinations += 1
            l_u = self.mention_network.node_data(node_u)
            l_v = self.mention_network.node_data(node_v)
            distance = round(haversine(l_u,l_v,miles=True),0)
            if distance > 10000:
                continue
            mentions_per_distance[distance] += 1.0
            self.N_squared += 1.0
            if self.mention_network.has_edge(node_u,node_v):
                following_relationship[distance] += 1.0
                self.S += 1.0

        x = list(sorted([key for key in mentions_per_distance]))
        x[0] += 1e-8
        y = []
        for key in mentions_per_distance:
            # "ratio of the number of pairs that have following relationship to the total number of pairs in the d_th bucket"
            mentions = mentions_per_distance[key]
            if mentions == 0:
                x.remove(key)
                continue
            following = following_relationship[key]
            ratio = following/mentions
            y.append(ratio)

        solutions = curve_fit(func_to_fit, x, y,p0=[-0.55,0.0045], maxfev=100000)[0]

        self.alpha = solutions[0]
        self.beta = solutions[1]
        return


    def generate_model_selector(self):
        for user in self.u_n:
            if np.random.binomial(1,
                                  self.F_r) == 1:  #generate a model selector, u according to a bernoulli distribution
                self.mu[user] = True
            else:
                self.mu[user] = False

            #normalizing K
            if np.random.binomial(1, (self.T_r[user] / self.K)) == 1:
                self.nu[user] = True
            else:
                self.nu[user] = False


    def random_following_model(self, user):
        """
        If mu = 1, we choose the random following model using p(f<i,j> == 1 | F_r)
        to decide if the location of a neighbor of the user is a possible location.
        """
        for neighbor in self.mention_network.neighbors_iter(user):
            if neighbor not in self.u_star:
                continue
            elif np.random.binomial(1, self.F_r):
                self.user_multi_locations[user].append(self.mention_network.node_data(neighbor))
        return


    def following_model(self, user):
        """
        If mu = 0, we decide whether there is f<i,j> based on the location-based following model as shown
        in eq. 1
        """
        #(note: this is almost the same as the Backstrom paper, thus I'll ignore generating
        #the theta values and just calculate max probability)
        def calculate_probability(l_u, l_v):
            """
            Calculates the probability, P(f<i,j>|alpha,beta,location_1,location_2)
            """
            try:
                return self.beta * (abs(haversine(l_u, l_v))) ** (self.alpha)
            except:
                #this needs to be changed to a very small value....
                return self.beta * (0.00000001) ** self.alpha

        best_log_probability = float('-inf')
        best_location = None
        for neighbor_u in self.mention_network.neighbors_iter(user):
            log_probability = 0
            if neighbor_u not in self.u_star:
                continue
            for neighbor_v in self.mention_network.neighbors_iter(neighbor_u):
                if neighbor_v not in self.u_star:
                    continue
                else:
                    l_u = self.mention_network.node_data(neighbor_u)
                    l_v = self.mention_network.node_data(neighbor_v)
                    plu_lv = calculate_probability(l_u, l_v)
                    try:
                        log_gamma_lu = math.log((plu_lv / (1 - plu_lv)))
                    except ValueError:
                        #in the case where l_u == l_v, then plu_lv --> 0 and log(1) = 0,
                        #thus this exception should be valid.
                        log_gamma_lu = 0
                    log_probability += log_gamma_lu
            if log_probability > best_log_probability:
                best_log_probability = log_probability
                best_location = self.mention_network.node_data(neighbor_u)
        if best_location:
            self.user_multi_locations[user].append(best_location)
        return


    def random_tweeting_model(self, user):
        for venue in self.mention_network.neighbors_iter(user):
            if venue not in self.venues:
                continue
            elif np.random.binomial(1, self.T_r[user]):
                self.user_multi_locations[user].append(self.mention_network.node_data(venue))
        return


    def tweeting_model(self, user):
        best_probability = float("-inf")
        best_venue = None

        for venue in self.mention_network.neighbors_iter(user):
            if venue not in self.venues:
                continue
            probability = self.psi[venue]
            if best_probability < probability:
                best_probability = probability
                best_venue = venue

        if best_venue:
            self.user_multi_locations[user].append(self.mention_network.node_data(best_venue))

        return


    def run_model(self):
        """
        run_model generates the values for all the initialized class variables, and
        follows the MLP algorithm described in the paper to infer locations for
        users.
        """

        #NOTE: K is not normalized to save computations, and is normalized on the fly in "generate_model_selector"
        #self.populate_mention_network()

        logger.debug("Variables have been initialized. Starting the model.")
        logger.debug("Storing location data...")
        self.store_location_data()
        self.u_n = self.nodes.difference(self.u_star)
        logger.debug("Location data stored!")


        logger.debug("Starting to compute the coefficients for the model...")
        #calculates the coefficients to be used in eq.1, alpha and beta
        self.compute_coefficients()
        logger.debug("Coefficients have been calculated. Alpha: %f and beta: %f." %(self.alpha, self.beta))

        logger.debug("Finding venue data..")
        self.find_locations()

        for venue in self.psi:
                self.psi[venue] /= self.K

        logger.debug("Finished finding venue data! %d venues found!" % len(self.venues))

        #self.N_squared = len(self.mention_network.edges_())
        #p(f<i,j> = 1 | F_r) = S / N^2
        self.F_r = (self.S / self.N_squared)

        #Section 4.4, generate model selector based on bernoulli distributions using T_r and F_r
        logger.debug("Generating model selectors...")
        self.generate_model_selector()
        logger.debug("Model selectors have been generated!")

        logger.debug("Starting to find user locations...")

        for user in self.u_n:
            if self.mu[user]:
                self.random_following_model(user)
            else:
                self.following_model(user)

            if self.nu[user]:
                self.random_tweeting_model(user)
            else:
                self.tweeting_model(user)

        logger.debug("Finished finding user locations...")

        for user in self.user_multi_locations:
            location_list = self.user_multi_locations[user]
            location = self.get_geometric_mean(location_list)
            self.mention_network.set_node_data(user,location)


    def return_network(self):
        return self.mention_network

    
    def get_geometric_mean(self, locations):
        """
        Locates the geometric mean of a list of locations, taken from David Jurgen's implementation,
        with less than three locations a random location is selected, else construct a geometric mean.
        """

        n = len(locations)

        # The geometric median is only defined for n > 2 points, so just return
        # an arbitrary point if we have fewer
        if n < 2:
            return locations[np.random.randint(0, n)]

        min_distance_sum = 10000000
        median = None  # Point type

        # Loop through all the points, finding the point that minimizes the
        # geodetic distance to all other points.  By construction median will
        # always be assigned to some non-None value by the end of the loop.
        for i in range(0, n):
            p1 = locations[i]
            dist_sum = 0
            for j in range(0, n):
                p2 = locations[j]
                # Skip self-comparison
                if i == j:
                    continue
                dist = haversine(p1, p2)
                dist_sum += dist

                # Short-circuit early if it's clear that this point cannot be
                # the median since it does not minimize the distance sum
                if dist_sum > min_distance_sum:
                    break

            if dist_sum < min_distance_sum:
                min_distance_sum = dist_sum
                median = p1

        return median
    



