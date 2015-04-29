##
#  Copyright (c) 2015, David Jurgens
#
#  All rights reserved. See LICENSE file for details
##

"""
A geoinference dataset is stored on disk in a directory with the following format:

	ds_root/
		dataset.json - metadata about the dataset
		posts.json.gz - all posts in the dataset in arbitrary order
		users.json.gz - all posts in the dataset grouped by user

This module provides a class for managing and accessing this directory as well
as helper functions for building new datasets.

**File formats**:

All three core files in the dataset directory contain data in JSON format.

  - `posts.json.gz` contains one post per line.  Each post is a JSON dictionary,
    the contents of git@drgitlab.cs.mcgill.ca:druths/sysconfig.gitwhich is specific to the platform the post was taken from.

  - `users.json.gz` contains one user per line.  Each user is a JSON dictionary
    with at least the following keys:
	  - `user_id` is a string identifier for the user that flags them as unique
	    among all other users in the dataset.
	  - `posts` is a JSON list which contains all the posts belonging to that user.
	    These posts should have identical format and data to those in the 
		`posts.json.gz` file.  Moreover, every post present in the `posts.json` file
		must be present in `users.json` and visa versa.
"""

import simplejson
import jsonlib
import os, os.path
import logging
import zen
import gzip
import subprocess

logger = logging.getLogger(os.path.basename(__file__))

class SparseDataset(object):
	"""
	This class encapsulates access to datasets.
	"""

	def __init__(self, dataset_dir, users_file='users.tsv.gz', excluded_users=set(), default_location_source='geo-median'):
		
		settings_fname = os.path.join(dataset_dir,'dataset.json')
		if os.path.exists(settings_fname):
			self._settings = jsonlib.load(open(settings_fname,'r'))
		else:
			self._settings = {}

		# prepare for all data
                self._dataset_dir = dataset_dir
		self._users_fname = os.path.join(dataset_dir, users_file)
		self._users_with_locations_fname = os.path.join(dataset_dir, 'users.home-locations.' + default_location_source + '.tsv.gz')
		self._mention_network_fname = os.path.join(dataset_dir, 'mention_network.elist')
		self._bi_mention_network_fname = os.path.join(dataset_dir, 'bi_mention_network.elist')
                self.excluded_users = excluded_users
		

	def post_iter(self):
		"""
		Return an iterator over all the posts in the dataset. The ordering
		of the posts follows the order of users in the dataset file.
		"""
		fh = gzip.open(self._users_fname,'r')

		for line in fh:
			user = self.load_user(line)
                        for post in user["posts"]:
                                yield post
                fh.close()

	def user_iter(self):
		"""
		Return an iterator over all posts in the dataset grouped by user. Each
		user is represented by a list of their posts - so any metadata about the
		user must be aggregated from the posts it produced.
		"""
		fh = gzip.open(self._users_fname,'r')

		for line in fh:
			user = self.load_user(line)

			yield user

	def __iter__(self):
		"""
		Return an iterator over all the posts in the dataset.
		"""
		return self.post_iter()

        def user_home_location_iter(self):
                """
                Returns an iterator over all the users whose home location has
                been already identified.  
                """
                location_file = self._users_with_locations_fname
                logger.debug('Loading home locations from %s' 
                             % (self._users_with_locations_fname))
                fh = gzip.open(location_file)
                logger.debug('Excluding locations for %d users' % (len(self.excluded_users)))
                for line in fh:
                        user_id, lat, lon = line.split('\t')
                        # print "%s %s" % (user_id, next(iter(self.excluded_users)))
                        if not user_id in self.excluded_users:
                                yield (user_id, (float(lat), float(lon)))
                        #else:
                        #        print "excluding %s" % user_id
                fh.close()
                        
                        
	def known_user_locations(self):
		"""
		Return dictionary of users to their locations, containing only
		users who have already self-reported their own location.  
		"""
		fh = gzip.open(self._users_fname,'r')

		for line in fh:
			user = self.load_user(line)
			yield user
                fh.close()


	def mention_network(self):
		"""
		Return the mention network for the dataset.
		"""
		return self.mention_network(bidirectional=False,directed=True,weighted=False)

	def bi_mention_network(self):
		"""
		Return the undirected mention network for the dataset consisting
		only of edges between users who have both mentioned each other.  This 
		"""
		return self.mention_network(bidirectional=True,directed=False,weighted=False)

	def build_graph(self,fname,directed,weighted):
		command = ("wc -l %s" %fname)
		process = subprocess.Popen(command, stdout=subprocess.PIPE,stderr=None, shell=True)
		output = process.communicate()
		graph_edge_capacity = int(output[0].split()[0]) + 1
		G = zen.edgelist.read(fname, weighted=weighted, ignore_duplicate_edges=True, merge_graph=zen.Graph(directed=directed, edge_capacity=graph_edge_capacity,edge_list_capacity=1))
		return G
		

	def mention_network(self, bidirectional=False, directed=False, weighted=False):
		"""
		Return the mention network for the dataset.
		"""
                if bidirectional:
                        if directed:
                                if weighted:
                                        fname = os.path.join(self._dataset_dir, 'bi_mention_network.directed.weighted.elist')
					return self.build_graph(fname,directed=True,weighted=True)
                                else:
                                        pass
                        else:
                                if weighted:
                                        fname = os.path.join(self._dataset_dir, 'bi_mention_network.weighted.elist')
					return self.build_graph(fname,directed=False,weighted=True)

                                else:
                                        fname = os.path.join(self._dataset_dir, 'bi_mention_network.elist')
					return self.build_graph(fname,directed=False, weighted=False)
                else:
                        if directed:
                                if weighted:
                                        pass
                                else:
                                        fname = os.path.join(self._dataset_dir, 'mention_network.elist')
					return self.build_graph(fname,directed=True, weighted=False)
                        else:
                                if weighted:
                                        pass
                                else:
                                        pass


        def load_user(self, line):
                """
                Converts this compressed representation of the user's data into
                a dict format that mirrors the full JSON data, except with all
                unused fields omitted (e.g., posting date).
                """
                cols = line.split("\t")
                user_id_str = cols[0]
                user_id = user_id_str
                posts = []
                user_obj = {} 
                user_obj['user_id']  = user_id
                user_obj['posts'] = posts
                COLS_PER_POST = 8

                should_exclude_location_data = user_id in self.excluded_users
                
                #print "User %d had line with %d columns (%f posts)" % (user_id, len(cols), len(cols) / 8.0)
                
                for post_offset in range(1, len(cols), COLS_PER_POST):
                        try:
                                # Grab the relevant content for this post
                                text = cols[post_offset]
                                tweet_id = cols[post_offset+1]
                                self_reported_loc = cols[post_offset+2]
                                geo_str = cols[post_offset+3]
                                mentions_str = cols[post_offset+4]
                                hashtags_str = cols[post_offset+5]
                                is_retweet_str = cols[post_offset+6]
                                place_json = cols[post_offset+7]                                                
                                
                                # Reconstruct the post as a series of nested dicts that
                                # mirrors the real-world full JSON object in structure
                                post = {}
                                post["id_str"] = tweet_id
                                post["id"] = long(tweet_id)
                                post["text"] = text

                                if is_retweet_str == 'True':
                                        # We don't have any data to put, so just fill it
                                        # with an empty object
                                        post["retweeted_status"] = {}

                                entities = {}
                                post["entities"] = entities

                                user_mentions = []
                                entities["user_mentions"] = user_mentions
                                if len(mentions_str) > 0:
                                        mentions = mentions_str.split(" ")
                                        for mention in mentions:
                                                mention_obj = {}
                                                mention_obj["id"] = long(mention)
                                                mention_obj["id_str"] = mention
                                                user_mentions.append(mention_obj)

                                hashtags = []
                                entities["hashtags"] = hashtags
                                if len(hashtags_str) > 0:
                                        tags = hashtags_str.split(" ")
                                        for tag in tags:
                                                tag_obj = {}
                                                tag_obj["text"] = tag
                                                hashtags.append(tag_obj)
                                
                        
                                # Only include geo information for posts that are not in
                                # the set of exlcuded posts, which are likely being used
                                # for testing data
                                if len(geo_str) > 0 and not should_exclude_location_data:
                                        geo = {}
                                        post["geo"] = geo
                                        coordinates = []
                                        coords = geo_str.split(" ")
                                        coordinates.append(float(coords[0]))
                                        coordinates.append(float(coords[1]))
                                        geo["coordinates"] = coordinates

                                # Place is a special case because the field formatting
                                # is so complex, it's just saved as a raw JSON string.
                                # This requires reparsing place to stuff in our object.
                                # However, since place is relative rare (1% of tweets),
                                # this isn't very expensive
                                if len(place_json) > 1 and not should_exclude_location_data:
                                        place = jsonlib.loads(place_json)
                                        post["place"] = place
                                user = {}
                                post["user"] = user
                                user["id_str"] = user_id_str
                                user["id"] = user_id
                                user["location"] = self_reported_loc

                                posts.append(post)
                        except:
                                logger.info("Saw malformed post when reading user; skipping")
                                pass
                        
                return user_obj
