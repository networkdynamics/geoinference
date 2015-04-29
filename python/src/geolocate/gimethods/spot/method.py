##
#  Copyright (c) 2015, Tyler Finethy
#
#  All rights reserved. See LICENSE file for details
##

"""
SPOT: Locating Social Media Users Based on Social Network Context

Author: Tyler Finethy
Date Created: September 2014
"""


from geopy.distance import vincenty
from geopy.distance import great_circle
from haversine import haversine
from geolocate import GIMethod, GIModel
from collections import defaultdict
from scipy.optimize import curve_fit
import numpy as np
import sys
import logging
import gzip
import os
try:
    import cPickle as pickle
except:
    import pickle

logger = logging.getLogger(__name__)

reload(sys)
sys.setdefaultencoding('utf-8')



class SpotMethod(GIMethod):
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
        SpotMethod.train_called = False
        SpotMethod.settings = None
        SpotMethod.model_dir = None
        SpotMethod.load_called = False
        SpotMethod.dataset = None

    def __init__(self):
        pass

    def train_model(self, settings, dataset, model_dir):
        """
        Creates a MultiLocationModel object classifier and saves the object as a pickled file.
        Settings should be in the format: {model : 'model', iterations : 'iterations'}, where
        model = "social_tightness_model" or "energy_localsocial_model", and iterations are the
        number of iterations to be performed.
        """
        SpotMethod.dataset = dataset

        #currently only supports social tightness
        spot_model = Spot_Social_Tightness()

        #running the model for 3 iterations
        spot_model.run_social_tightness_model(3)

        #getting the altered mention network from the spot_model
        network = spot_model.return_network()

        user_id_to_location = {}
        for user_id in network.nodes_iter():
            try:
                location = network.node_data(user_id)
                if not location is None:
                    user_id_to_location[user_id] = location
            except KeyError:
                pass            

        model = SpotModel(user_id_to_location)

        if model_dir:
            logger.debug("Storing the model in %s" % model_dir)
            SpotMethod.model_dir = model_dir
            filename = os.path.join(model_dir, 'user-to-lat-lon.tsv.gz')

            fh = gzip.open(filename, 'w')
            for user_id, loc in user_id_to_location.iteritems():
                fh.write("%s\t%f\t%f\n" % (user_id, loc[0], loc[1]))
            fh.close()


        return model
        pass

    def load_model(self, model_dir, settings=None):
        """
        Loads a MultiLocationModel object classifier from a pickled file.
        """
        self.load_called = True
        self.model_dir = model_dir
        if settings:
            self.settings = settings
	fh = gzip.open(os.path.join(model_dir, 'user-to-lat-lon.tsv.gz'), 'r')
        user_id_to_location = {}
        for line in fh:
            cols = line.split("\t")
            user_id_to_location[cols[0]] = (float(cols[1]), float(cols[2]))
        fh.close()

        return SpotModel(user_id_to_location)



class SpotModel(GIModel):
    """
    MultiLocationModel extends GIModel and is the classifier for geo-inferring tweets.
    """
    num_posts_inferred = 0
    num_users_inferred = 0

    @staticmethod
    def clear():
        SpotModel.num_posts_inferred = 0
        SpotModel.num_users_inferred = 0


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
        SpotModel.num_posts_inferred += 1
        userID = post['user']['id_str']
        if SpotModel.num_posts_inferred % 10000000 == 0:
            logger.debug('Spot has inferred the location of %d posts' % (SpotModel.num_posts_inferred))
        try:
            return self.user_id_to_location[userID]
        except KeyError:
            return None

    def infer_posts_by_user(self, posts):
        """
        Returns a list of locations given multiple tweets from a single user
        """
        SpotModel.num_users_inferred += 1
        return [self.infer_post_location(post) for post in posts]


class Spot(object):
    """
    Spot Object to be extended by the Social Tightness
    and Energy and Local Social Coefficient Models.
    Classes that extend the Spot Object must initialize
    the self.mention_network, self.users_with_location, and
    the self.nodes objects.
    """
    def __init__(self):
        """
        The classes that inherit Spot will have a self.mention_network
        that is a Zen-Network based on the dataset object passed to the Method.
        """
        logger.debug("Initializing the SPOT method")
        self.mention_network = SpotMethod.dataset.bi_mention_network()
        self.users_with_location = []
        self.users_without_location = []
        self.nodes = set(self.mention_network.nodes())
        self.sij = {}

        # run populate mention network to assign known location to users,
        # also fills the users_with_location and users_without_location
        self.store_location_data()

    def return_network(self):
        """
        Returns the mention_network with inferred location of users
        """
        return self.mention_network

    def store_location_data(self):
        """
        Sets the node_data field with the relevant gold-standard location data from
        the bidirectional dataset.
        """
        num_users_seen = 0
        for user_id, loc in SpotMethod.dataset.user_home_location_iter():
            if loc[0] == 0 and loc[1] == 0:
                continue
            try:
                self.mention_network.set_node_data(user_id, loc)
                self.users_with_location.append(self.mention_network.node_idx(user_id))
                num_users_seen += 1
                if num_users_seen % 100000 == 0:
                    logger.debug('UserProfiling saw %d users' % num_users_seen)
            except KeyError:
                pass
        tmpset = set(self.users_with_location)

class Spot_Social_Tightness(Spot):
    """
    SPOT: Locating Social Media Users Based on Social Network Context
    """
    def run_social_tightness_model(self,iterations):
        """
        For each iteration performed by run_social_closeness,
        first the social_closeness and probability_distance_social_closeness
        dictionary is built. Then the most probable location for users are set
        """

        #the set of users with found location
        #this will be cleared after the users_with_location/
        #users_without_location sets are updated
        logger.debug("Starting Spot: Social Tightness model for %d iterations.."% iterations)
        self.users_with_found_location = set()
        for n in range(iterations):
            logger.debug("Iteration #%d" %(n+1))
            
            for user in self.users_with_found_location:
                self.users_with_location.append(user)
            
            self.users_without_location = set(self.mention_network.nodes_()).difference(set(self.users_with_location))   
            
            self.users_with_found_location.clear()

            # "We obtain the probability p(|li -lj|,sij) of user ui and uj located at
            #  the distance of |li-lj| with social closeness sij from  training data."
            # this is initialized and built on each iteration
            self.probability_distance_social_closeness = defaultdict(lambda: defaultdict(float))
            self.sij = defaultdict(lambda:defaultdict(float))
            self.social_closeness()

            self.estimate_user_location()
        logger.debug("Social Tightness model completed...")


    def social_closeness(self):
        """
        The social tightness based model is based on the assumption
        that different friends hae different importance to a user.
        The social closeness between two users is measured via cosine similarly,
        then we estimate the probability of user i and user j located at a distance
        | l_i - l_j | with social closeness. Then we estimate the probability of user_i
        located at l_i and use the location with the top probability
        """
        pairs = 0
        #here we calculate social closeness
        logger.debug("Calcuating social closeness")
        for user in self.users_with_location:
            user_location = self.mention_network.node_data_(user)
            for friend in self.mention_network.neighbors_iter_(user):
                friend_location = self.mention_network.node_data_(friend)
                if not friend_location: continue

                pairs += 1
                social_closeness = round(self.cosine_similarity(user,friend),2)
                self.sij[user][friend] = social_closeness
                distance = round(haversine(user_location,friend_location),0)
                self.probability_distance_social_closeness[distance][social_closeness] += 1.0

        #the normalizing factor is the total number of social_closeness probabilities added above...
        normalizing_factor = pairs
        for distance in self.probability_distance_social_closeness:
            for social_closeness in self.probability_distance_social_closeness[distance]:
                self.probability_distance_social_closeness[distance][social_closeness] /= normalizing_factor
        logger.debug("Finished calculating the social closeness...")

    def estimate_user_location(self):
        #estimates the probability of user_i located at li
        #uses the location with the top probability
        users = 0
        locations = 0
        logger.debug("Beginning to estimate user location...")
        location_set = set(self.users_with_location)
        for user in self.users_without_location:
            if users % 1000000 == 0:
                logger.debug("Found a location for %d users of %d" %(locations,users))
            users+=1
            likely_location = self.most_likely_location(user,location_set)
            self.mention_network.set_node_data_(user,likely_location)
            if likely_location:
                locations += 1
                self.users_with_found_location.add(user)


    def most_likely_location(self,user,location_set):
        """
        Returns the most likely location for a user of unknown locale,
        based on the social tightness model.
        """
        max_probability = float('-inf')
        best_location = None
        for neighbor_u in self.mention_network.neighbors_iter_(user):
            if neighbor_u not in location_set: continue

            location_of_neighbor_u = self.mention_network.node_data_(neighbor_u)
            probability = 0

            for neighbor_v in self.mention_network.neighbors_iter_(neighbor_u):
                if neighbor_v not in location_set: continue
                location_of_neighbor_v = self.mention_network.node_data_(neighbor_v)

                #to get the dict-lookup correct, we round to the nearest kilometer
                distance = round(haversine(location_of_neighbor_u,location_of_neighbor_v),0)

                # "" , round to two significant figures
                social_closeness = self.sij[neighbor_u][neighbor_v]
                probability += self.probability_distance_social_closeness[distance][social_closeness]

            #compare the probability of this neighbor with other possible neighbors
            #sets the highest-probability user as the most likely location
            if probability > max_probability:
                max_probability = probability
                best_location = location_of_neighbor_u
    
        return best_location


    def cosine_similarity(self,user,friend):
        """
        Returns cosine similarity between a user in a friend measured as,
        sij = |(set of friends of user i) intersected (set of friends of user j)| divided by
         sqrt(#of friends of user i * # of friends of user j)
        """
        #counting the intersection of friends
        intersection = 0
        #counting the neighbors of the user
        user_neighbors = self.mention_network.degree_(user)
        #counting the neighbors of the friend
        friend_neighbors = self.mention_network.degree_(friend)

        #avoid division by zero error
        if not user_neighbors or not friend_neighbors:
            return 0.0

        if user_neighbors >= friend_neighbors:
            for friend_neighbor in self.mention_network.neighbors_iter_(friend):
                if self.mention_network.has_edge_(friend_neighbor,user):
                    intersection += 1
        else:
            for user_neighbor in self.mention_network.neighbors_iter_(user):
                if self.mention_network.has_edge_(user_neighbor,friend):
                    intersection +=1

        cos_similarity = float(intersection) / np.sqrt(user_neighbors**2 * friend_neighbors**2)

        return cos_similarity


class Spot_Energy(Spot):
    """
    Spot_Energy extends the Spot Class, initializing
    the zen network, self.mention_network, and the sets
    self.nodes, self.user_with_location and self.user_without_location.

    Spot_Energy is an implementation fo the Energy and Local
    Social Coefficient Model
    """
    def run_spot_energy_model(self,iterations=2):
        """
        Performs the Spot Energy Model algorithm, adding
        the inferred location of users with initially unknown
        location over various iterations.

        One iteration is necessary for training
        """
        trained = False

        #this coefficients will be trained on first iteration
        self.alpha, self.beta1, self.beta2


        self.users_with_found_location = set()
        for n in range(iterations):
            for user in self.users_with_found_location:
                self.users_with_location.add(user)
                self.users_without_location.remove(user)

            self.users_with_found_location.clear()
            # social closeness between two users measured by cosine similarity,
            # this will be initialized and built on each iteration
            self.sij = defaultdict(lambda:defaultdict(float))

            #builds self.sij and self.probability_distance_social_closeness
            self.build_social_closeness()

            self.dr = defaultdict(float)

            #user u_i at l_i the energy generated by his/her friend u_j
            self.g_ui = defaultdict(float)
            self.energy_generated()

            #c is defined as the social coefficient,
            # C(user) = 3 * G_triangle / 3 * G_triangle + G_open triples
            self.c = defaultdict(float)
            self.social_coefficients()

            self.rg_rc = defaultdict(list)


            if not trained:
                self.best_neighbor = {}
                self.training_build_rc_rg()
                self.alpha, self.beta1, self.beta = self.logistic_response_function_coefficients()
                trained = True

            else:
                self.build_rc_rg()
                self.infer_location()

    def infer_location(self):
        """
        TODO
        """
        for user in self.users_without_location:
            if self.rg_rc[user]:
                location = max([(self.logistic_response_function(rg,rc),location) for (rg,rc,location) in self.rg_rc[user]])[0][1]
                self.mention_network.set_node_data(location)
                self.users_with_found_location.add(user)


    def build_rc_rg(self):
        """
        TODO
        """
        for user in self.users_without_location:
            rgs = []
            rcs = []
            for neighbor in self.mention_network.neighbors(user):
                if not neighbor in self.users_with_location:
                    continue
                location = self.mention_network.node_data(neighbor)
                rg = self.g_ui[neighbor]
                rc = self.c[neighbor]
                rgs.append((rg,location))
                rcs.append((rc,location))
            rgs = sorted(rgs)
            rcs = sorted(rcs)

            for rg in rgs:
                #storing the ranking index number r_G, and r_C, and the location
                self.rg_rc[user].append((rgs[1].index(rg[1]) + 1,rcs[1].index(rg[1]) + 1,rgs[1]))

    def logistic_response_function(self,rg,rc):
        """
        TODO
        """
        return np.exp(self.alpha + self.beta1*rg + self.beta2*rc)/(1+np.exp(self.alpha+self.beta1*rg+self.beta2*rc))

    def training_build_rc_rg(self):
        """
        TODO
        """
        for user in self.users_with_location:
            rgs = []
            rcs = []
            for neighbor in self.mention_network.neighbors(user):
                if not neighbor in self.users_with_location:
                    continue
                location = self.mention_network.node_data(neighbor)
                rg = self.g_ui[neighbor]
                rc = self.c[neighbor]
                rgs.append((rg,location))
                rcs.append((rc,location))
            rgs = sorted(rgs)
            rcs = sorted(rcs)

            user_location = self.mention_network.node_data(user)
            best_distance = float('inf')
            for neighbor_location in rg:
                if vincenty(neighbor_location[1],user_location).km < best_distance:
                    self.best_neighbor[user] = neighbor_location[1]

            for rg in rgs:
                #storing the ranking index number r_G, and r_C, and the location
                self.rg_rc[user].append((rgs[1].index(rg[1]) + 1,rcs[1].index(rg[1]) + 1,rgs[1]))


    def logistic_response_function_coefficients(self):
        def func_to_fit((rg,rc),alpha,beta1,beta2):
            return np.exp(alpha + beta1*rg + beta2*rc)/(1+np.exp(alpha+beta1*rg+beta2*rc))
        x = []
        y = []
        for user in self.rg_rc:
            for value in self.rg_rc[user]:
                if value[2] == self.best_neighbor[user]:
                    x.append((value[0],value[1]))
                    y.append(1)
                else:
                    x.append((value[0],value[1]))
                    y.append(0)

        solutions = curve_fit(func_to_fit,x,y)[0]
        self.alpha = solutions[0]
        self.beta1 = solutions[1]
        self.beta2 = solutions[2]
        return


    def build_social_closeness(self):
        """
        builds the sij vector
        """
        #here we calculate social closeness
        for user in self.users_with_location:
            for friend in self.users_with_location:
                if friend == user: continue
                self.sij[user][friend] = self.cosine_similarity(user,friend)


    def cosine_similarity(self,user,friend):
        """
        Returns cosine similarity between a user in a friend measured as,
        sij = |(set of friends of user i) intersected (set of friends of user j)| divided by
         sqrt(#of friends of user i * # of friends of user j)
        """
        #counting the intersection of friends
        intersection = 0
        #counting the neighbors of the user
        user_neighbors = 0
        #counting the neighbors of the friend
        friend_neighbors = 0
        for neighbor_user in self.mention_network.neighbors_iter(user):
            user_neighbors += 1.0
            for neighbor_friend in self.mention_network.neighbors_iter(friend):
                friend_neighbors += 1.0
                if neighbor_friend == neighbor_user: intersection += 1.0

        #avoiding division by zero error
        if user_neighbors == 0 or friend_neighbors == 0:
            return 0

        cos_similarity = intersection / np.sqrt(user_neighbors * friend_neighbors)

        return cos_similarity


    def energy_generated(self):
        """
        The total energy of u_i located at l_i is:
        G(u_i,l_i) = -1 * sum of sij * g<u_i,u_j> over all friends.
        """
        #building dr, making ten bins based on social similarity
        distances_by_social_similarity = defaultdict(list)
        for user in self.users_with_location:
            location_user = self.mention_network.node_data(user)
            for neighbor in self.mention_network.neighbor_iter(user):
                if user == neighbor:
                    continue
                location_neighbor = self.mention_network.node_data(neighbor)
                social_similarity_rounded = round(self.sij[user][neighbor],1) #rounded to one significant figure
                distance = vincenty(location_user,location_neighbor)
                distances_by_social_similarity[social_similarity_rounded].append(distance)
        for social_similarity in distances_by_social_similarity:
            distances = distances_by_social_similarity[social_similarity]
            self.dr[social_similarity] = sum(distances)/len(distances)


        for user in self.users_with_location:
            location_user = self.mention_network.node_data(user)
            for neighbor in self.mention_network.neighbors_iter(user):
                if not user in self.users_with_location:
                    continue
                location_neighbor = self.mention_network.node_data(neighbor)

                social_similarity = self.sij[user][neighbor]
                #the exponent term, g<u_i,u_j> = -e^(-|l_i - l_j|/d_r)
                x = - vincenty(location_user,location_neighbor) / self.dr[round(social_similarity,1)]
                #summing sij * g<u_i,u_j> over all friends
                #I've factored out a -1 from np.exp(x) and cancelled it with
                #the leading -1 in the summation.
                self.g_ui[user] += social_similarity* np.exp(x)


    def social_coefficients(self):
        """
        Calculates C(U_i), the social coefficient for each user
        with known location.
        """
        for user in self.users_with_location:
            #the number of friends, defined as gamma_i
            neighbors = set(self.mention_network.neighbors(user))
            gamma_i = len(neighbors)
            g_open_triplets = gamma_i * (gamma_i - 1)/ 2
            g_closed_triplets = 0
            while neighbors:
                neighbors, g_closed_triplets = g_closed_triplets(neighbors,g_closed_triplets)
            self.c[user] = (3 * g_closed_triplets)/(3*g_closed_triplets + g_open_triplets)


    def closed_triplets(self,neighbors,g_closed_triplets):
        """
        Calculates the amount of closed triplets for a user,
        it works by taking the set of all a users neighbors,
        popping the
        """
        neighbor = neighbors.pop()
        neighbors_neighbors = set(self.mention_network.neighbors(neighbor))
        intersection = neighbors.intersection(neighbors_neighbors)
        neighbors = neighbors.difference(intersection)
        g_closed_triplets += len(intersection)
        return neighbors, g_closed_triplets



















