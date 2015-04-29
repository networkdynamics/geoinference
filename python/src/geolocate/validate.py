##
#  Copyright (c) 2015, Yi Tian Xu
#
#  All rights reserved. See LICENSE file for details
##

import sys
import timeit
import abc
import argparse
import json

from haversine import haversine

sys.path.insert(0, '../')
from geolocate import dataset
from geolocate import geocoder

def iter_user_post(dataset_obj, mode):
	# mode = true if by_user
	if mode:
		for user_posts in dataset_obj.user_iter():
			for post in user_posts['posts']:
				yield (post['id'], post['geo'])
	else:
		for post in dataset_obj.post_iter():
			yield (post['id'], post['geo'])
		

def validate(dataset_dir, result_file, mode='dist'):
	'''
		mode = 'dist'
		Check the distance error between the inferred location and the geotag (if any)
		Get mean and median distance error

		mode = 'name-city', 'name-reg', 'name-cty'
		Check if the name of the location matches at a certain level of granularity 
	'''

	res = {}
	if mode == 'dist':
		res['dist'] = []
		res['mean'] = 0.0
	else:
		res['match'] = 0.0
		res['gg'] = geocoder.Geocoder()
		if mode == 'name-city':
			res['gran'] = 0
		elif mode == 'name-reg':
			res['gran'] = 1
		else:
			res['gran'] = 2

	count = set_counter("validated posts")

	D = dataset.Dataset(dataset_dir)
	with open(result_file, 'r') as result:
		l = result.readlines()
		read_mode = json.loads(l[0])['by_user']
		lines = l[1:]
		print 'start validating'
		for i, val in enumerate(iter_user_post(D, read_mode)):
			_id, geo = val 
			coord = []
			# check if post has geotag
			if geo:
				coord = geo['coordinates']
				# check if geotag has valid coordinate
				if not valid_coord(coord):
					geo = None
			n = count.update(geo)
			if not geo: continue
			_id_res, lat_res, lon_res = lines[i].split('\t')
			true_coord = map(float, [lat_res, lon_res])			

			# check if the id is the same
			assert int(_id_res) == _id
			
			if mode == 'dist':
				d = haversine(true_coord, coord)
				res['dist'].append(d)
				res['mean'] += (d-res['mean'])/n
			else:
				d = res['gg'].reverse_geocode(true_coord[0], true_coord[1])
				if d: d = d[res['gran']]
				pre_d = res['gg'].reverse_geocode(coord[0], coord[1])
				if pre_d: pre_d = pre_d[res['gran']]
				n = count.downdate(d and pre_d)
				if not d or not pre_d: continue
				m = 0
				if d == pre_d:
					m = 1
				res['match'] += (m-res['match'])/n


	count.output()
	if mode == 'dist':
		return {'mean' : mean, 
			'median' : sorted(res['dist'])[len(res['dist'])/2]}
	else:
		return res['match']

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

# counter class to visualize the number when needed (mainly in the gimethods)

class counter:
	def __init__(self):
		self.start = timeit.default_timer()

	@abc.abstractmethod
	def update(self, a):
		pass

	def write(self, s):
		sys.stdout.write("\b"*len(s)) 
		sys.stdout.flush()
		sys.stdout.write("%s" % s)
		sys.stdout.flush()
 
	def output(self):
		print timeit.default_timer() - self.start, 'sec'

class set_counter(counter):
	'''
		used if you want to count the elements in a subset
		i.e.: given a set S, it counts the number of elements in S and the number 
		of elements in S satisfying a certain condition A

		this can be used for the training set, for example, if we want to show 
		how many posts has geotags, and thus can be trained
	'''
	def __init__(self, mess = '', step = 1024):
		self.message = mess
		self.counted = 0
		self.total = 0
		self.step = step
		counter.__init__(self)

	def update(self, a):
		self.total += 1
		if a:
			self.counted += 1
		try:
			if self.total % self.step < 1:
				self.write(str(self.counted) + ' / ' + str(self.total))
		except:
			pass
		return self.counted
	
	def downdate(self, a):
		if not a:
			self.counted -= 1
		return self.counted	

	def output(self):
		self.write(str(self.counted) + ' / ' + str(self.total))
		print ' -- ', self.counted*100.0/self.total, '%', self.message
		counter.output(self)

class progress(counter):
	'''
		counts the number of element processed
	'''
	def __init__(self, mess='', step = 32):
		counter.__init__(self)
		self.step = step
		self.mess = mess

	def update(self, a):
		if a%self.step == 0:
			self.write(str(a))

	def output(self, a):
		self.write(str(a))
		print " ", self.mess


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset_dir', help='directory for the dataset folder')
	parser.add_argument('result_dir', help='path to the inference output file')
	parser.add_argument('mode', choices=['dist', 'name-city', 'name-reg', 'name-cty'], help='validation according to location coordinates or names')
	
	args = parser.parse_args()

	print validate(args.dataset_dir, args.result_dir, args.mode)

if __name__ == "__main__":
	main()
