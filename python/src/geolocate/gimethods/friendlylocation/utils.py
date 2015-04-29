##
#  Copyright (c) 2015, Yi Tian Xu, James McCorriston, David Jurgens
#
#  All rights reserved. See LICENSE file for details
##

import sys
import timeit
import re
import math

# import requests
import numpy as np
from haversine import haversine


def valid_coord(coord):
    if coord[0] < -90 or coord[0] > 90 or coord[1] < -180 or coord[1] > 180:
        return False
    return True

def isCoord(loc):
    text = re.sub(u'([^0-9]+)', u'', loc)
    if len(text) > 10:
        text = re.sub(u'([^0-9]+)', u' ', loc)
        token = text.split()
        raw_coord = loc.replace(',', ' ').replace(':', ' ')
        coord = []
        if len(token) == 4:
            coord = to_float(raw_coord.split())
        elif len(token) in (6, 8):
            coord = to_dec(token, raw_coord)
        if len(coord) == 2 and valid_coord(coord):
            return coord
    return None

# return distance in miles
def distance(loc1, loc2):
	if loc1 == None or loc2 == None:
		return -1
	return haversine(loc1, loc2, miles=True)

# get random coordinates
def rand_coord():
	coord = np.random.random(2)
	coord = np.array([coord[0]*180-90, coord[1]*360 - 180])
	return coord

# the great circle distance between a user's home location (from their tweets) and the location returned from the geocoder
# i.e.: distance between geotags (true_loc) and location field (text_loc) 
def location_error(true_loc, coord, LocRes):
	# we create location resolver in method.py because we don't want it to load every time we import this file
	if not true_loc: return 0.0
	# check if location field contains coordinates
	#coord = isCoord(text_loc)
	if coord: return haversine(true_loc, coord)
	# resolve to lat lon 
	res = LocRes.reverse_geocode(text_loc.split()[0],text_loc.split()[1])
	if not res: return 0.0
	res_val = map(float, res)
	return haversine(true_loc, res_val)

# create a vector 
# [ mention relationship, location error, post data, social triangles ]
def contact_vector(params):
	post_data, graph_data, loc_err, st_count = params
	mnt_to, mnt_back = graph_data
	#if int(mnt_to) + int(mnt_back) > 2:
		#print 'graph_data: ' + str(graph_data)
	vector =  [ #is_friend, is_follower, int(is_friend and is_follower),	
			    mnt_to, mnt_back, #int(mnt_to and mnt_back),
			    loc_err if loc_err else 0.0, # mean location error
			    #post_data.pop('fnd_count', 0), # number of friends
			    #post_data.pop('flw_count', 0), # number of followers
			    #post_data.pop('acc_type', 2),  # public or private account
			    st_count] # number of edge in social triangle
	return np.array(vector)

# the curve form of p
def curve_form(dist, a, b, c):
	return a*np.power(dist+b,-c)


def centroid(points):
	# by average
	return np.mean(np.array(points), axis=0).tolist()

# check if user has at least t geotags 
def is_geocoded(user_posts, t):
	geotags = []
	for post in user_posts['posts']:
	# check if coordinates exist
		if 'geo' in post and post['geo']:
			coord = map(lambda x: float(x), post['geo']['coordinates'])
			if valid_coord(coord):
				geotags.append(coord)
	if len(geotags) >= t:
		return centroid(geotags)
	return None

# get number of friends, follower, and the account type (private or public)	
def get_post_data(user_posts):
	for post in user_posts: 
		try:
			loc = post['user']['location']
		except:
			loc = ''
		return ( loc, { #'fnd_count' : post['user']['friends_count'],
			        #'flw_count' : post['user']['followers_count'],
			        #'acc_type' : int(post['user']['protected'])
				} )

# find the parameters necessary to compute the location error for each post
# average the error
def get_loc_error(user_post):
	mle = 0.0
	count = 0
	for post in user_posts['posts']:
		loc = post['location']
		if not loc:
			return 0.0
		try:
			coord = post['user']['coordinates']
		except:
			continue
		if lr.valid_coord(coord):
			count += 1
			mle += (location_error(loc, coord) - mle)/count
	return mle

