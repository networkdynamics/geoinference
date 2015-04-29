##
#  Copyright (c) 2015, Tyler Finethy
#
#  All rights reserved. See LICENSE file for details
##

"""
Twitter geolocation inference model based on:
'Towards Social User Profiling: Unified and Discriminative Influence Model for Inferring Home Locations'

Author: Tyler Finethy
Date Created: June 2014
"""

from numpy.random import uniform as un
import os, os.path
import logging
import time
import gzip

from collections import defaultdict
from geolocate import GIMethod, GIModel
from geolocate import geocoder

#from twokenize import tokenizeRawTweetText as tokenizer

try:
    import cPickle as pickle
except:
    import pickle

logger = logging.getLogger(__name__)


class UserProfilingMethod(GIMethod):
    """
    UserProfilingMethod extends GIMethod
    """
    train_called = False
    settings = None
    dataset = None
    model_dir = None
    load_called = False

    @staticmethod
    def clear():
        UserProfilingMethod.train_called = False
        UserProfilingMethod.settings = None
        UserProfilingMethod.dataset = None
        UserProfilingMethod.model_dir = None
        UserProfilingMethod.load_called = False

    def __init__(self):
        pass

    def train_model(self,settings,dataset,model_dir=None):
        """
        Creates a UserProfilingModel object classifier and saves the object as a pickled file
        """
        UserProfilingMethod.train_called = True
        UserProfilingMethod.settings = settings
        UserProfilingMethod.dataset = dataset

        logger.debug("Starting UserProfilingMethod")
        network =  UserProfiling(settings, UserProfilingMethod.dataset).return_network()

        user_id_to_location = {}
        for user_id in network.nodes_iter():
            try:
                location = network.node_data(user_id)
                if not location is None:
                    user_id_to_location[user_id] = location
            except KeyError:
                pass            
        model = UserProfilingModel(user_id_to_location)

        logger.debug("Finished UserProfilingMethod! Generated a model!")


        if model_dir:
            logger.debug("Storing the model in %s" %model_dir)

            filename = os.path.join(model_dir, 'user-to-lat-lon.tsv.gz')
            fh = gzip.open(filename, 'w')
            for user_id, loc in user_id_to_location.iteritems():
                fh.write("%s\t%s\t%s\n" % (user_id, loc[0], loc[1]))
            fh.close()


        return model


    def load_model(self,model_dir,settings=None):
        """
        Loads a UserProfilingModel object classifier from a pickled file object
        """
        self.load_called = True
        self.model_dir = model_dir
        if settings:
            self.settings = settings

        user_id_to_location = {}
        model_file = gzip.open(os.path.join(model_dir, "user-to-lat-lon.tsv.gz"), 'r')
        for line in model_file:
            line = line.strip()
            cols = line.split("\t")
            user_id = cols[0]
            lat = float(cols[1])
            lon = float(cols[2])
            user_id_to_location[user_id] = (lat, lon)

        logger.debug("Loaded locations for %d users" % len(user_id_to_location))

        model_file.close()
        return UserProfilingModel(user_id_to_location)


class UserProfilingModel(GIModel):
    """
    UserProfilingModel extends GIModel
    """
    num_posts_inferred = 0
    num_users_inferred = 0

    @staticmethod
    def clear():
        UserProfilingModel.num_posts_inferred = 0
        UserProfilingModel.num_users_inferred = 0

    def __init__(self, user_id_to_location):
        """
        Initializes the class based on the model/classifier created by UserProfilingMethod
        """
        self.user_id_to_location = user_id_to_location
        return

    def infer_post_location(self,post):
        """
        Infers the class given the text from a tweet or post.
        """
        UserProfilingModel.num_posts_inferred += 1
        user_id = post['user']['id_str']
        if user_id in self.user_id_to_location:
            return self.user_id_to_location[user_id]
        else:
            return None

    def infer_posts_by_user(self,posts):
        """
        Infers classes given the text from multiple tweets or posts.
        """
        UserProfilingModel.num_users_inferred += 1
        return [self.infer_post_location(post) for post in posts]


class UserProfiling(object):
    def __init__(self, settings, dataset):

        if 'location_source' in settings:
            self.geocoder = geocoder.Geocoder(dataset=settings['location_source'])
        else:
            self.geocoder = geocoder.Geocoder()

        logger.debug("Loading mention network...")
        self.mention_network = dataset.bi_mention_network()
        logger.debug("Starting to store the location data in the mention network...")
        self.U = set()
        self.store_location_data()
        logger.debug("Finished storing the location data of users in the mention network")        

        #how accurate the *_old dictionary values are to the new values (this will be a deciding factor in runtime).
        self.CONVERGENCE_ERROR = 0.1

        #dictionaries on dictionaries... contains users known location, calculated locations, last location,
        #probably a much better way to do this but for now I'll leave it like this...
        logger.debug("Initializing dictionaries...")
        self.X = defaultdict(float)
        self.Y = defaultdict(float)
        self.X_old = defaultdict(float)
        self.Y_old = defaultdict(float)
        self.X_step = defaultdict(float)
        self.Y_step = defaultdict(float)
        self.X_step_old = defaultdict(float)
        self.Y_step_old = defaultdict(float)

        #sigma values for each node
        self.sigma = defaultdict(float)

        #the set of users with unknown location, as described in the paper
        self.U_n = set()
        #the set of users with known location
        self.U_star = set()

        for node in self.mention_network.nodes_iter():
            self.U.add(node)
            if self.mention_network.node_data(node) == None:
                self.X[node] = un(-90,90)
                self.Y[node] = un(-180,180)
                self.X_step[node] = -1*self.X[node]
                self.Y_step[node] = -1*self.Y[node]
                self.U_n.add(node)
            else:
                self.X[node] = self.mention_network.node_data(node)[0]
                self.Y[node] = self.mention_network.node_data(node)[1]
                self.U_star.add(node)

        self.locations = set()

        logger.debug("Starting to find locations")
        self.find_locations()
        logger.debug("Found locations! %d" % len(self.locations))

        logger.debug("Starting global prediction algorithm")
        self.global_prediction_algorithm()
        logger.debug("Finished global predictiona algorithm!")

        for location in self.locations:
            self.mention_network.rm_node(location)


    def find_locations(self):
        users_seen = 1
        for possible_posts in UserProfilingMethod.dataset.user_iter():
            users_seen += 1
            if users_seen % 1000000 == 0:
                logger.debug("Seen %d users" %users_seen)
            user_id = possible_posts['user_id']
            if user_id in self.U_n:
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
        self.locations.add(location_name)
        self.X[location_name] = location[0]
        self.Y[location_name] = location[1]
        if self.mention_network.has_edge(user_id, location_name):
            self.mention_network.set_weight(user_id, location_name, self.mention_network.weight(user_id, location_name) + 1)
        else:
            self.mention_network.add_edge(user_id, location_name, 1)


    def global_prediction_algorithm(self):
        """
        while the sum of the difference of the current lat/lon versus the next step has yet to converge,
        the global prediction algorithm runs.
        """

        #outer iteration as described in the paper, Algorithm 1: "Global Prediction Algorithm"
        loops = 1
        logger.debug("Starting outer convergence loop")
        while self.convergence_outer() and loops < 6:
            logger.debug("Outer Convergence Loop Number: %d" %loops)
            loops += 1

            for node in self.U:
                self.sigma_users(node)

            for location in self.locations:
                self.sigma_locations(location)

            #inner_iteration
            checked = True
            inner_loops = 1
            while self.convergence_inner() or checked:
                logger.debug("Inner Convergence Loop Number: %d" % inner_loops)
                inner_loops += 1
                checked = False
                for node in self.U_n:
                    if self.sigma[node] == 0.0:
                        continue
                    self.compute_location(node)

            for node in self.U_n:
                self.X_old[node] = self.X[node]
                self.Y_old[node] = self.Y[node]
                self.X[node] = self.X_step[node]
                self.Y[node] = self.Y_step[node]

        #set final locations for unknown-location nodes
        for node in self.U_n:
            self.mention_network.set_node_data(node,(self.X[node],self.Y[node]))
        return

    def sigma_users(self,node):
        """
        equation 13 as described in the paper,
        properly sets the sigma value for each neighbor in the network
        """
        total = 0.0
        number_of_friends = 0
        for friend in self.mention_network.neighbors_iter(node):
            if friend in self.U:
                total += ((self.X[friend] - self.X[node])**2 + (self.Y[friend] - self.Y[node])**2)
                number_of_friends += 1

        if number_of_friends == 0:
            return

        total /= (2.0*number_of_friends)

        self.sigma[node] = total
        return

    def sigma_locations(self,location):
        """
        equation 14 as described in the paper,
        properly sets the sigma value for each location
        """
        weight_total = 0
        total = 0
        for node in self.mention_network.neighbors_iter(location):
            if location in self.mention_network.neighbors(node):
                weight = self.mention_network.weight(node,location)
                weight_total += weight
                total += weight*(((self.X[node] - self.X[location])**2 + (self.Y[node] - self.Y[location])**2))
        if weight_total == 0:
            return 0
        total /= (2.0*weight_total)
        self.sigma[location] = total
        return

    def compute_location(self,node):
        """
        equation 12 as described in the paper
        """
        numerator_x = 0.0
        numerator_y = 0.0
        denominator = 0.0
        for neighbor in self.mention_network.neighbors_iter(node):
            if neighbor in self.U:
                if self.sigma[neighbor] == 0.0 or self.sigma[neighbor] == 0.0:
                    continue
                numerator_x += (self.X[neighbor]/(self.sigma[node])) + (self.X[neighbor]/self.sigma[neighbor])
                numerator_y += (self.Y[neighbor]/(self.sigma[node])) + (self.Y[neighbor]/self.sigma[neighbor])
                denominator += (1.0/self.sigma[node]) + (1.0/self.sigma[neighbor])
            if neighbor in self.locations:
                if self.sigma[neighbor] == 0.0:
                    continue
                numerator_x += (self.mention_network.weight(node,neighbor)*self.X[neighbor])/self.sigma[neighbor]
                numerator_y += (self.mention_network.weight(node,neighbor)*self.Y[neighbor])/self.sigma[neighbor]
                denominator += self.mention_network.weight(node,neighbor)/self.sigma[neighbor]

        self.X_step_old[node] = self.X_step[node]
        self.Y_step_old[node] = self.Y_step[node]
        if denominator == 0:
            self.X_step[node] = self.X_step_old[node]
            self.Y_step[node] = self.Y_step_old[node]
            return
        self.X_step[node] = (numerator_x/denominator)
        self.Y_step[node] = (numerator_y/denominator)
        return

    def convergence_inner(self):
        """
        Checks for convergence of the next-step iteration, returns True while
        next step value hasn't converged.
        """
        for node in self.U_n:
            if abs(self.X_step[node] - self.X_step_old[node]) > self.CONVERGENCE_ERROR:
                return True
            elif abs(self.Y_step[node] - self.Y_step_old[node]) > self.CONVERGENCE_ERROR:
                return True
        return False

    def convergence_outer(self):
        for node in self.U_n:
            if abs(self.X[node] - self.X_old[node]) > self.CONVERGENCE_ERROR:
                return True
            elif abs(self.Y[node] - self.Y_old[node]) > self.CONVERGENCE_ERROR:
                return True
        return False

    def store_location_data(self):
        """
        Sets the node_data field with the relevant gold-standard location data from
        the bidirectional dataset.
        """
        num_users_seen = 0
        for user_id, loc in UserProfilingMethod.dataset.user_home_location_iter():
            if loc[0] == 0 and loc[1] == 0:
                continue
            try:
                self.mention_network.set_node_data(user_id, loc)
                num_users_seen += 1
                if num_users_seen % 100000 == 0:
                    logger.debug('UserProfiling saw %d users' % num_users_seen)
            except KeyError:
                pass


    def return_network(self):
        """
        Returns the zen network with userIDs, edges, and most importantly lat/longs for all users
        """
        return self.mention_network
