##
#  Copyright (c) 2015, David Jurgens
#
#  All rights reserved. See LICENSE file for details
##

import os.path
import logging
import gzip
import pickle

from collections import Counter

from geolocate import GIMethod, GIModel, geocoder


LOGGER = logging.getLogger(os.path.basename(__file__))

class Davis_Jr_et_al_Model(GIModel):

    def __init__(self, user_id_to_location):
        self.user_id_to_location = user_id_to_location

    def infer_post_location(self, post):
        if not "user" in post:
            return None
        user = post["user"]

        if not "id" in user:
            return None

        user_id = user["id"]
        
        #LOGGER.debug("searching for id %s" % user_id)

        # If we know this user's location, report their home location
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


class Davis_Jr_et_al_Method(GIMethod):

    def __init__(self):
        # Location is represented as a lat/lon geopy Point
        self.user_id_to_location = {}
        self.geocoder = None;
        self.geo = 0
        self.loc = 0
        self.geoip = 0
        self.user_to_home_loc = {}


    def train_model(self, settings, dataset, model_dir):
        """
        Infers the locations of users from the gold standard data available and
        then predicts the locations of unknown users from their friend's
        locations
        """

        # Initialize the geocoder, which we'll use to resolve location strings.
        # We use the default name-to-location mapping unless the user has
        # specified otherwise.
        can_use_home_loc = True
        if 'location_source' in settings and not settings['location_source'] == 'geo-median':
            self.geocoder = geocoder.Geocoder(dataset=settings['location_source'])
        else:
            self.geocoder = geocoder.Geocoder()
            can_use_home_loc = False
        
        min_location_votes = 2
        if 'min_location_votes' in settings:
            min_location_votes = int(settings['min_location_votes'])

        min_friends = 1
        if 'min_friends' in settings:
            min_friends = int(settings['min_friends'])

        # Arbitrarily large upper bound...
        max_friends = 100000
        if 'max_friends' in settings:
            max_friends = int(settings['max_friends'])

        # Most recent posts to use
        posts_to_use = 10
        if 'posts_to_use' in settings:
            posts_to_use = int(settings['posts_to_use'])


        LOGGER.debug('Loading mention network')
        mention_network = dataset.bi_mention_network()

        # For each of the users that we have in the network, see if we can
        # associate that user with a home location.
        all_users = set(mention_network.nodes_iter())
        for user_id, home_loc in dataset.user_home_location_iter():           
            if not user_id in all_users:
                continue
            self.user_to_home_loc[user_id] = home_loc

        # This dict will contain a mapping from a user ID to an associated
        # location, if one can be discovered from the GPS, GeoIP, or Location
        # Field data.
        user_to_gold_loc = {}
        num_users_seen = 0
        
        LOGGER.debug('Inferring home locations of %s users' % len(all_users))
        for user in dataset.user_iter():
            user_id = user["user_id"]
            # Skip the posts of users who aren't in the mention network.  This
            # is potentially inefficient, but there's currently no support for
            # iterating by user-id.
            if not user_id in all_users:
                continue

            # This strategy returns null if no location was found
            home_loc = self.get_location(user["posts"], posts_to_use, can_use_home_loc)
            if not home_loc is None:
                user_to_gold_loc[user_id] = home_loc
                #last_seen_user = user_id
            num_users_seen += 1
            if num_users_seen % 100000 == 0:
                LOGGER.debug('Seen %d/%d users, located %d (%d geo, %d geo-ip, %d loc field)' % (num_users_seen, len(all_users), len(user_to_gold_loc), self.geo, self.geoip, self.loc))
                #LOGGER.debug("check for %s" % last_seen_user)
                #break
        
        # Once we have a gold-standard set of locations, infer the locations of
        # all other users on the basis of their friends
        for user_id in all_users:
            if user_id in user_to_gold_loc:
                self.user_id_to_location[user_id] = user_to_gold_loc[user_id]
                continue

            #LOGGER.debug("Testing %s" % user_id)

            neighbors = mention_network.neighbors(user_id)
            
            # Check that the user's ego network is within the prescribed bounds
            if len(neighbors) < min_friends or len(neighbors) > max_friends:
                #print "Outside friends bounds %d" % (len(neighbors))
                continue

            # For each of the users in the network, get their estimated
            # location, if any
            locationCounts = Counter()
            for neighbor_id in neighbors:
                in_gold = str(neighbor_id in user_to_gold_loc)
                #LOGGER.debug("%s -> %s ? %s" % (user_id, neighbor_id, in_gold))
                if neighbor_id in user_to_gold_loc:
                    locationCounts[user_to_gold_loc[neighbor_id]] += 1

            # Skip this user if we didn't have any locatable friends
            if len(locationCounts) == 0:
                #print "No locatable friends"
                continue
            
            # Choose the most common, arbitrarily breaking ties.
            most_common_loc = locationCounts.most_common(1)[0]
            
            # Ensure that at least the minimum number of friends had this
            # location
            if locationCounts[most_common_loc] < min_location_votes:
                #print "Not enough votes"
                continue

            #print "Made it through %s" % user_id
            
            
            #LOGGER.debug("%s -> %s" % (user_id, str(most_common_loc)))
            self.user_id_to_location[user_id] = most_common_loc


        LOGGER.debug('Inferred home locations of %d users' % len(self.user_id_to_location))

        if not model_dir is None:
            #fh = open(os.path.join(model_dir, 'davis-jr-model.pickle'), 'w')
            #pickle.dump(self.user_id_to_location, fh)
            #fh.close()
            
            # Write the .tsv for human debugability too
            fh = gzip.open(os.path.join(model_dir, 'user-to-lat-lon.tsv.gz'), 'w')
            for user_id, loc in self.user_id_to_location.iteritems():
                fh.write("%s\t%s\t%s\n" % (user_id, loc[0], loc[1]))
            fh.close()

        return Davis_Jr_et_al_Model(self.user_id_to_location)


    def get_location(self, posts, posts_to_use, can_use_home_loc):
        """
        Identifies a settlement location from the GPS, GeoIP, or self-reported
        location field of the author's posts, returning that location or None if
        the user did not provide any of those three forms of information
        """

        # We don't need all the posts
        if len(posts) > posts_to_use:
            posts = posts[-posts_to_use:]

        # First, check for GPS data
        for post in posts:
            if not "geo" in post:
                continue
            geo = post['geo']
            if not "coordinates" in geo:
                continue
            coords = geo["coordinates"]
            lat = coords[0]
            lon = coords[1]

            # Convert the point to its city name and then re-convert that city
            # back to a lat-lon.  This normalizes the lat-lon data to a single
            # point for a city.
            self.geo += 1
            return self.geocoder.canonicalize(lat, lon)

        # Next, check for GeoIP city location
        for post in posts:
            if not "place" in post:
                continue
            place = post["place"]
            if not "place_type" in place:
                continue
            place_type = place["place_type"]

            # We only want to deal with city places; all others are too hard to
            # resolve
            if not place_type == "city":
                continue

            # Get a name out of this city
            location_name = place["full_name"] + "\t" + place["country"]
            
            # Convert the name to our canonical lat/lon for it.
            #
            # NOTE: we should probably check that lat/lon is within the bounding
            # box that Twitter provides, otherwise it's an error and we should
            # pick somethign like the mid-point of the bounding box.
            self.geoip += 1
            return self.geocoder.geocode(location_name)

        # Finally, check for a self-reported.  There's no need to iterate over
        # all the posts for this, since the self-reported location is mostly
        # constant.
        if not "user" in post:
            return None

        user = post["user"]
        

        if can_use_home_loc:
            if not "id" in user:
                return None

            user_id_str = user["id_str"]

            # See if we've extracted a home_location for this user
            if not user_id_str in self.user_to_home_loc:
                return None
            lat_lon = self.user_to_home_loc[user_id_str]
        else:
            location_name = user['location']
            # Otherwise, we'll have to ask the geocoder to look for the location
            # for us
            lat_lon = self.geocoder.geocode_noisy(location_name)       

        if not lat_lon is None:
            self.loc += 1

        return lat_lon

        


    def load_model(self, model_dir, settings):
        """
        Reads in the pickled Davis Jr. et al model
        """      

        fh = gzip.open(os.path.join(model_dir, 'user-to-lat-lon.tsv.gz'), 'r')
        user_id_to_location = {} 
        for line in fh:
            cols = line.split("\t")
            user_id_to_location[cols[0]] = (float(cols[1]), float(cols[2]))
        fh.close()
        return Davis_Jr_et_al_Model(user_id_to_location)
