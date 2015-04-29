##
#  Copyright (c) 2015, Yi Tian Xu, James McCorriston, David Jurgens
#
#  All rights reserved. See LICENSE file for details
##

import math
import json
from collections import defaultdict, Counter
import abc
from operator import itemgetter
import timeit
import random
import time
import os
import gzip

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib
from scipy import optimize

import logging

import utils
from geolocate import GIMethod, GIModel
from geolocate.dataset import Dataset
from geolocate.validate import set_counter
from geolocate.geocoder import Geocoder
import os
import os.path

LOGGER = logging.getLogger(__name__)

# notations:
# L = locations set of the target's contacts 
# P = predicted distances set between target and contacts (used for inferred user)
# D_a = actual distances between targets and contacts
# D_p = predicted distances between targets and contacts
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

class network:
	# this call shall be extended by the model and the method

	def __init__(self):
		# this is the relationship network
		# but we only have the mention graph for now
		self.X = None

	def get_user(self, user_id):
		# get node data
		data = self.X.node_data(user_id)
		if not data:
			data = {}
		return data

	def get_contacts(self, user_id, direction = False): 
		# get all neighbors in both direction
		#for n in self.X.in_neighbors_iter(user_id):
		#	if direction:
		#		yield ('in', n)
		#	else:
		#		yield n
		#for n in self.X.out_neighbors_iter(user_id):
		#	if direction:
		#		yield ('out', n)
		#	else:
		#		yield n
		for n in self.X.neighbors_iter(user_id):
			yield n

	def get_contact(self, user_id, cont): 
		# get edge data
		data = self.X.edge_data(user_id, cont)
		if not data:
			data = {}
		return data

	def get_user_data(self, user_id, data):
		# to get a certain type of node data
		try: 
			return self.get_user(user_id)[data]
		except:
			return None

	def get_contact_data(self, user_id, contact, data):
		# to get a certain type of edge data
		try:
			return self.get_contact(user_id, contact)[data]
		except:
			pass
		try:
			return self.get_contact(contact, user_id)[data]
		except:
			pass

	def get_d_a(self, user_id, cont):
		# get the true distance between user and contact
		return self.get_contact_data(user_id, cont, 'd_a')

	def get_l_a(self, user_id):
		# get the true location of user
		return self.get_user_data(user_id, 'l_a')

	def get_social_triangles(self, user_id, cont):
		return self.get_contact_data(user_id, cont, 'st')

	def get_relationship(self, user_id, contact):
		# get the relationship types between user and contact
		# we only have foward mention and backward mention for now
		graph_data = [0]*2
		if self.X.has_edge(user_id, contact):
			graph_data[0] = self.X.weight(user_id, contact)
		if self.X.has_edge(contact, user_id):
			graph_data[1] = self.X.weight(contact, user_id)
		return graph_data

	def get_params(self, _id, c):
		# extract all information needed to create the contact vector 
		# mentions
		graph_data = self.get_relationship(_id, c)
		# number of social triangles (based on mentions)
		st_count = self.get_social_triangles(_id, c)
		if not st_count:   
			st_count = self.social_triangles(_id, c)
			data = self.get_contact(_id, c)    
			data['st'] = st_count
			self.X.set_edge_data(_id,c,data)  
		# friend count, follower count, account type
		post_data = self.get_user_data(c, 'pd')
		if not post_data:
			post_data = {}
		# location error
		loc_err = self.get_user_data(c, 'le')
		return (post_data, graph_data, loc_err, st_count)

	def get_contact_vector(self, *args):
		return utils.contact_vector(self.get_params(*args))

	def social_triangles(self, user_id, contact):
		# find the number of mentions that are common between user and contact
		count = 0 
		cnt = self.get_contacts(contact)
		for m in self.get_contacts(user_id):
			if m in cnt:
				count += 1
		return count 


class FriendlyLocation_Model(GIModel, network):
	def __init__(self, user_to_loc):
		self.user_to_loc = user_to_loc
		#print 'len user_to_loc:'
		#print len(self.user_to_loc)
		
		
	def infer_post_location(self, post):
		# pretend that we are inferring for a user 
		# and call infer_posts_locations_by_user
		_id = post['user']['id']
		return self.infer_posts_locations_by_user(_id, [post])[0]

	def infer_posts_locations_by_user(self, user_id, posts):
		if user_id in self.user_to_loc:
			return [self.user_to_loc[user_id]]*len(posts)
		else:
			return [None]*len(posts)



class FriendlyLocation(GIMethod, network):
	def __init__(self):
		self.dist = {}
		network.__init__(self)
		 # the connection zen graph in the format
		''' node object: user_id 
			node data : {
					'pd': data from the posts
					'le': location error
					'l_a': true location
				    }
			edges data : {
					'd_a': true distance between target and contact
					'st': social triangles in common 
				     } 
			edge weight : 0 for foward mention, 1 for follower, 2 for friend '''
		# note that only mention is supported for now
		# to avoid repeated target-contact pair (edge uv and edge vu), we only store edges data 
		# on one directions
		self.user_to_home_loc = {}
		self.stgrEdges = Counter() # hash the stgrEdge values 
		self.actEdges = {}
		self.actEdgesTuples = []
		self.hit = 0
		#self.method = kwargs.pop('method', 'stranger')
		self.method = 'stranger'
		self.qntl = {}
		self.P = {} # predicted locations of the contacts from a particular user

	def add_user_data(self, _id, coord, post_data):
		try:
			self.X.set_node_data(_id, { 'l_a': coord, 'pd': post_data})
		except KeyError:
			pass

	# compute all distance between users with true locations
	def set_d_a_for_all(self):
		for u,v in self.X.edges_iter():
			if self.get_contact_data(v,u,'d_a'):
				continue 
			dist = utils.distance(self.get_l_a(u), self.get_l_a(v))
			# if no l_a
			if dist < 0: continue
			data = {}
			data['d_a'] = dist
			self.X.set_edge_data(u,v,data)

	# helper method to set location error
	def set_loc_err(self, _id, loc_err):
		data = self.get_user(_id)
		data['le'] = loc_err
		self.X.set_node_data(_id, data)

	# helper method to set the true distance
	def set_d_a(self, target, contact, d_a):
		data = self.get_contact(u, v) 
		data['d_a'] = d_a
		self.X.set_edge_data(target, contact, data)

	# only iter through contacts that have d_a
	def iter_contacts(self):
		for u,v in self.X.edges_iter():
			if self.get_d_a(u, v):
				yield (u, v)

	# local contact ratio: the mean distance from the user's contact
	def LCR(self, user):
		# LRC(l, L) = number of contacts whose distance is smalled than self.min_dist (default is 25 miles) / 
		# 		number of contacts 
		count = [0.0, 0.0]
		for c in self.get_contacts(user):
			dist = self.get_d_a(c)
			if c:
				count[0] += 1 if dist < self.min_dist else 0
				count[1] += 1
		return count[0]/count[1]

	# find the quantile boundaries 
	def quantile_boundaries(self, X):
		# q_i = D_[1+jn/m] for i<m
		# q_m = infinity
		# where D is the set if all distance 
		ind = 0
		# iter through all pairs that have a distance 
		for u, c in self.iter_contacts():
			self.dist[(u, c)] = (self.get_d_a(u, c), self.predict(X[ind])[0])
			ind += 1

		D_p = sorted([d[1] for _, d in self.dist.iteritems()])
		n = len(D_p)*1.0
		#print [D_p[int(j*n/self.m)] for j in xrange(self.m)]
		#print [int(j*n/self.m) for j in xrange(self.m)]
		self.set_qntl_bound([D_p[int(j*n/self.m)] for j in xrange(self.m)])
		self.qntl[10] = [float('inf'), ()]


	# the number of edges that belong to a quantile and have distance of d miles between target and contact
	def getActEdges(self):
		for j in range(0, self.m+1):
			self.actEdges[j] = Counter()
		for u1, u2 in self.actEdgesTuples:
			try:
				d_a, d_p = self.dist[(u1,u2)]
			except KeyError:
				try:
					d_a, d_p = self.dist[(u2,u1)]
				except KeyError:
					#print u1 + ' ' + u2
					continue
			d_a = round(d_a, 1)
			#if d - d_a < 1e-8:
			#print 'd_p'
			#print d_p
			j = self.qntl_map(d_p)
			#print 'j'
			#print j
			self.actEdges[j][d_a] += 1
			#self.allActEdges[d_a] += 1
		fact = open('act_edges.tsv', 'w')
		for j in range(0, self.m+1):
			for distance in self.actEdges[j]:
				fact.write(str(j) + '\t' + str(distance) + '\t' + str(self.actEdges[j][distance]) + '\n')
		fact.close()


	# the number of edges that could have existed at a distance d
	"""
	def getStgrEdges(self):
		count = 0
		for loc in self.X.nodes_iter():
			count += 1
			#if count % 100 == 0:
			print 'Count: %d' % count
			#for u2 in self.X.grp_out_neighbors_iter([u for u in self.X.nodes_iter()]):
			for u2 in self.X.nodes_iter():
				if u is not u2:
					distance = round(utils.distance(self.get_l_a(u), self.get_l_a(u2)), 1)
					if distance in self.stgrEdges_:
						self.stgrEdges_[distance] += 1
					else:
						self.stgrEdges_[distance] = 1
		print 'dumping'
		json.dump(stgrEdges, open('stranger_edges.json', 'w'))
	"""

	# probability that a contact in a quantile j lives d miles form the target user
	def p(self, j, d):
		if self.stgrEdges[d] == 0:
			return 0.0
		return float(self.actEdges[j][d])/self.stgrEdges[d]

	# calculates p(d) for d in d_lst	
	def get_p(self, d_lst, j):
		ps = {}
		for d in set(d_lst):
			ps[d] = self.p(j, d)
		g = np.array([ps[d] for d in d_lst])
		return g

	# using optimization to fit the curve form of the probability
	def fit_curves(self, d_lst):
		#print 'getting stranger edges'
		#self.getStgrEdges()
		print 'getting actual edges'
		self.getActEdges()
		for j in xrange(self.m+1):
			p_lst = self.get_p(d_lst, j)
			#LOGGER.debug(j, p_lst, d_lst)
			popt,pcov = optimize.curve_fit( utils.curve_form, 
							d_lst, p_lst, maxfev=100000)
			#LOGGER.debug('done', j)
			self.set_qntl_prob(j, popt)

	def reset(self):
		# called after inference to prepare to infer another user
		self.P = {}

	def iter_contacts_inf(self):
		# iter through the contacts of the inferring user
		for c in self.P:
			yield (self.get_l_a(c), self.P[c])

	### set methods for quantiles

	def set_qntl_bound(self, arr):
		# prepare to setup the quantile bounds
		for i, q in enumerate(arr):
			self.qntl[i] = [q, ()]

	def set_qntl_prob(self, j, p):
		# set quantile j to probability p
		try:
			self.qntl[j][1] = p
		except:
			self.qntl[j] = [None, p]
		#print self.qntl

	### get methods for quantiles

	def get_qntl_bound(self, j): return self.qntl[j][0]

	def get_qntl_prob(self, j): return self.qntl[j][1]

	
	### methods to call the classifier

	def predict(self, X): 
		return self.tree.predict(X)

	def fit(self, X,y): self.tree.fit(X,y)

	# mapping of a predicted distance to a quantile	
	def qntl_map(self, d_p):
		for j, _ in enumerate(self.qntl):
			if d_p < self.get_qntl_bound(j):
				return j
		return len(self.qntl)-1

	
	def p_inf(self, j, dist):
		# p*(j,d) = aj(bj+d)^{-cj}
		return utils.curve_form(dist, *self.get_qntl_prob(j))

	def nearest_neighbor(self):
		# find nearest neighbor with a location 
		sorted_P = sorted(self.P.iteritems(), key=itemgetter(1),  reverse=True)
		for c in self.P:
			coord = self.get_l_a(c)
			if coord:
				return coord
		return [0,0]

	'''
		FriendlyLocation formula:
			log(fl(l, L, P)) = log(D(l, L, P)) + log(pStgrs(l))
			
			where P is the set of predicted distances from user to contact 
			and L is the location of the contacts

			log(D(l, L, P)) = sum_k[p(qntl(p_k), dist(l, l_k) - (1 - p(dist(l, l_t)))]
			for k is a contact of the user we are inferring
			
			and 
			
			log(pStgrs(l)) = sum_t[1 - p(dist(l, l_t))]
			for t is a user in the network

			so 
			log(fl(l, L, P)) = sum_k[p(qntl(p_k), dist(l, l_k)] + sum_[t != k] [1 - p(dist(l, l_t))]
	'''

	def pStgrs(self, l, user_id):
		# TODO: pSTgrs seems to always give the same number, if this is correct, then we can save and reuse the value 
		res = 0.0
		neighborhood = self.X.in_neighbors(user_id) + self.X.out_neighbors(user_id)
		for u in self.X.nodes_iter():
			if u in neighborhood:
				continue
			coord = self.get_l_a(u)
			if coord:
				dist = utils.distance(coord, l)
				res += 1-self.p_inf(self.qntl_map(dist), dist)
		return res

	#IMPORTANT: Using the version of the FL equation that ignores the pStrangers to save on computation time.
	#Returns the negated log of FL
	def fl(self, l, user_id):
		# if input coordinates is not within valid range
		lloc = l
                #print str(lloc)
		while not valid_coord(lloc):
			if lloc[0] > 90:
				lloc = (lloc[0] - 180, lloc[1])
			elif lloc[0] < -90:
				lloc = (lloc[0] + 180, lloc[1])
			if lloc[1] > 180:
				lloc = (lloc[0], lloc[1] - 360)

			elif lloc[1] < -180:
				lloc = (lloc[0], lloc[1] + 360)

		#if not valid_coord(l):
			#return 63700 # circumference of earth x 10
		res = 0.0
		
		# this part is D(l, L, P)
		#print 'fl'
		for l_k, p_k in self.iter_contacts_inf():
			if l_k is not None:
				dist = utils.distance(lloc, l_k)
				j = self.qntl_map(p_k)
				if self.stgrEdges[dist] > 0 and self.actEdges[j][dist] > 0:
					p = float(self.actEdges[j][dist]) / self.stgrEdges[dist]
				else:
					p = 0
				#p = self.p_inf(j, dist)
				#print 'p'
				#print p
					#print 'p'
					#print p
					#print 'logp'
					#print np.log(p)
				if p > 0:
					res += np.log(p)
			# Add the next 7 lines to add the denominator back into the FL equation
			#try:
				#p2 = self.allActEdges[dist]/self.stgrEdges[dist]
				#print p2
			#except Exception as e:
				#print e
				#p2 = 0
			#res -= np.log(1 - p2)
		# pStgrs(lloc)
		# Add the next 2 lines to add the pStgrs back into the FL equation
		#if self.method == 'default':
			#return res + self.pStgrs(lloc, user_id)
		#print 'res: '
		#print res
		if res == 0.0:
			return 1000000
		return -res

	#given the list of all known neighbour locations (ls), determine which of these
	#returns the maximum value for FL (function in the paper).
	def find_max_fl(self, user_id, ls):
		minp = float('inf')
		maxloc = None
		prob = 0
		for loc in ls:
			#print 'neighbour loc:'
			#print loc
			prob = self.fl(loc, user_id)
			#print 'prob:'
			#print prob
			if prob < minp:
				minp = prob
				maxloc = loc
		return maxloc

	def infer_locs(self):
		user_to_loc = {}
		c = 1
		start = time.time()
		for user_id in self.X.nodes_iter():
			#if the user already has a known location, we do not need to infer it
			if user_id in self.user_to_home_loc:
				user_to_loc[user_id] = self.user_to_home_loc[user_id]
			else:
				#get the locations of each of the user's neighbours
				start = time.time()
				ls = []
				for m in self.get_contacts(user_id, True):
					vec = self.get_contact_vector(user_id, m)
					#print 'vec'
					#print vec
					self.P[m] = self.predict(vec)[0]
					#print 'val'
					#print self.P[m]
					l_a = self.get_l_a(m)
					if l_a is not None:
						ls.append(l_a)
			
				#the user must have at least one friend with a known location in order to infer their location
				if len(ls) > 0:
					# nearest neighbor
					if self.method == 'nearest':
						result = self.nearest_neighbor()*len(posts)
					else:
						# default or stranger
						#minimization used since FL returns the negative log of it's probability value
						#types = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'SLSQP', 'dogleg', 'trust-ncg']
						temp = utils.rand_coord()
						#for typ in types:
						start = time.time()
						#res = optimize.minimize(lambda x: self.fl(x, user_id), temp, method = typ)
						res = self.find_max_fl(user_id, ls)
						#print 'found max fl: ' + str(time.time() - start)
						#print typ + ': ' + str(list(res.x))
					result = res
					#print result
					start = time.time()
					while not valid_coord(result):
						#print 'INVALID'
						#print result
						if result[0] > 90:
							result = (result[0] - 180, result[1])
						elif result[0] < -90:
							result = (result[0] + 180, result[1])
						if result[1] > 180:
							result = (result[0], result[1] - 360)
						elif result[1] < -180:
							result = (result[0], result[1] + 360)
				#if the user has no friends with a known location
				else:
					result = None
				
				user_to_loc[user_id] = result
				# reset self.P
				self.reset()
				#if any(result):
				#if result is not None:
					#LOGGER.debug('Result: (' + str(result[0]) + ', ' + str(result[1]) + ')')
			if c%1000 == 0:
				#print time.time() - start
				start = time.time()
			c += 1
		return user_to_loc

	def train_model(self, settings, dataset, model_dir=None):
		# settings in the form 
		'''{ 'LCR_min_dist' : the cutoff distance to distinguish between local and non-local contacts (default = 40 km ~ 25 miles)
			 'qntl_num' : the number of quantiles (default is 10)
			 'min_geotag' : the minimum number of geotags that makes a user a target (deafault = 3)
			 'min_samples_leaf' : the maximum number of sample in a leaf of the regression tree, i.e: the minimum for the regressor to not split a leaf (default = 1000)
		   }'''
		self.min_dist = settings.pop('LCR_min_dist', 40)
		self.m = settings.pop('qntl_num', 10)
		self.min_geotag = settings.pop('min_geotag', 3)
		min_samp_leaf = settings.pop('min_samples_leaf', 1000)
		LOGGER.debug('tree')
		self.tree = DecisionTreeRegressor(min_samples_leaf=min_samp_leaf) # the classifier
		LOGGER.debug('geocoder')
		#LocRes = Geocoder()
                if 'location_source' in settings:
                    LocRes = Geocoder(dataset=settings['location_source'])
                else:
                    LocRes = Geocoder()


		LOGGER.debug('loading mention network')
		self.X = dataset.mention_network(bidirectional=True, directed=True, weighted=True)
		#print len(self.X)
		#counter = set_counter('has at least %d geotags'%self.min_geotag) ### counter
		# adding users
		self.user_to_home_loc = {user: loc for (user, loc) in dataset.user_home_location_iter()}
		user_loc_list = self.user_to_home_loc.items()
		random.shuffle(user_loc_list)
		#print len(user_loc_list)
		#Take a sample from user home locations to estimate stgrEdges and actEdges
		start = time.time()
		user_loc_list = user_loc_list[:50000]
		#print 'home loc time:'
		#print len(user_loc_list)
		#fstgr = open('stgr_edges.tsv', 'w')
		c = 0
		LOGGER.debug('sampling stranger edges and actual edges')
		for uid1, loc1 in user_loc_list:
			#if c % 100 == 0:
			#	print c
			c2 = 0
			for uid2, loc2 in user_loc_list:
				if not c2 == c:
					if self.X.has_edge(uid1, uid2):
						self.actEdgesTuples.append((uid1, uid2))
					distance = round(utils.distance(loc1, loc2), 1)
					self.stgrEdges[distance] += 1
				c2 += 1
			c += 1
		#for distance in self.stgrEdges:
		#	fstgr.write(str(distance) + '\t' + str(self.stgrEdges[distance]) + '\n')
		#fstgr.close()
		#print len(self.actEdgesTuples)
		LOGGER.debug('filling network')
		for _id, loc in dataset.user_home_location_iter():
			#_id = user['user_id']
			#loc = UserProfilingMethod.dataset.user_home_location_iter()
			#loc, pd = utils.get_post_data(user['posts'])
			#l_a = utils.is_geocoded(user, self.min_geotag)
			#counter.update(loc) ### counter
			#if not self.X.__contains__(_id):
				#self.X.add_node(_id)
			if loc[0] == 0 and loc[1] == 0:
				continue
			else:
				try:
					self.X.add_node(_id)
				except:
					pass
			l_a = loc
				#if not l_a:	continue
			self.add_user_data(_id, l_a, {})				
			le = utils.location_error(l_a, loc, LocRes)
			self.set_loc_err(_id, le)

			# remove mentions of itself
			if self.X.has_edge(_id, _id):
				self.X.rm_edge(_id, _id)

		LOGGER.debug(str(self.X.__len__()) + 'users')
		LOGGER.debug(str(self.X.size()) + 'edges')

		self.set_d_a_for_all()

		tempx = []
		tempy = []
		for u, x in self.iter_contacts():
			tempx.append(self.get_contact_vector(u,x))
			tempy.append(self.get_d_a(u,x))
		X = np.array(tempx)
		Y = np.array(tempy)
		#X = np.array([self.get_contact_vector(u,x)
		#				for u, x in self.iter_contacts()])
		#Y = np.array([self.get_d_a(u,x)
		#				for u, x in self.iter_contacts()])

		LOGGER.debug('number of relationships' + str(len(X)))

		LOGGER.debug("fitting"	)	
		start = timeit.default_timer() 
		#try:
		self.fit(X, Y)
		#except:
		#	raise RuntimeError, 'No connections to train on.'
	
		LOGGER.debug('done fitting tree -' + str(timeit.default_timer() - start) + 'sec')

		start = timeit.default_timer()
		self.quantile_boundaries(X)
		LOGGER.debug('done setting quantile boundaries -' + str(timeit.default_timer() - start) + 'sec' )

		start = timeit.default_timer()
		self.fit_curves(Y)
		LOGGER.debug('done fitting curves -' + str(timeit.default_timer() - start) + 'sec' )
		
		#self.model.allActEdges = self.allActEdges
		#self.model.stgrEdges = self.stgrEdges
		
		self.user_to_loc = self.infer_locs()
		if model_dir is not None:
			LOGGER.debug('saving model')
			filename = os.path.join(model_dir, "user-to-lat-lon.tsv.gz")
			fh = gzip.open(filename, 'w')
			for user_id, loc in self.user_to_loc.iteritems():
                        	if not loc is None:
					fh.write("%s\t%s\t%s\n" % (user_id, loc[0], loc[1]))
			fh.close()

			
		self.model = FriendlyLocation_Model(self.user_to_loc)
		return self.model

	def load_model(self, model_dir, settings=None):
		path_settings = os.path.join(model_dir, "user-to-lat-lon.tsv.gz")
		user_to_loc = {}
		fh = gzip.open(path_settings, "r")
		for line in fh:
		    cols = line.split("\t")
		    user_to_loc[cols[0]] = (float(cols[1]), float(cols[2]))
		fh.close()
		return FriendlyLocation_Model(user_to_loc)
