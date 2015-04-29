##
#  Copyright (c) 2015, Derek Ruths, David Jurgens
#
#  All rights reserved. See LICENSE file for details
##

from geolocate.dataset import *
import zen
import unittest
import os, os.path
import gzip

post_file = os.path.join(os.path.dirname(__file__),'posts.json.gz')

class DatasetBuildingTestCase(unittest.TestCase):

	def test_build_users(self):
		posts2users(post_file,lambda x: str(x['user_id']))

		# check the users file
		users_file = os.path.join(os.path.dirname(__file__),'users.json.gz')
		posts = [json.loads(x) for x in gzip.open(post_file,'r')]
		users = [json.loads(x) for x in gzip.open(users_file,'r')]

		# confirm that there is the correct number of users
		self.assertEquals(len(users),7)
		
		# confirm that there is the correct number of posts
		num_posts = sum(map(lambda x: len(x['posts']),users))
		self.assertEquals(num_posts,len(posts))
		
		# delete the users file
		os.remove(users_file)

	def test_build_mention_network(self):
		posts2mention_network(post_file,lambda x: str(x['user_id']),
							  lambda x: map(str,x.get('mentions',[])))

		# check the network file
		network_fname = os.path.join(os.path.dirname(__file__),'mention_network.elist')
		self.assertTrue(os.path.exists(network_fname))

		# load the network
		G = zen.edgelist.read(network_fname,directed=True,weighted=True)

		self.assertEquals(len(G),6)
		self.assertEquals(G.size(),5)

		# done
		os.remove(network_fname)

	def test_build_dataset(self):
		dataset_dir = os.path.join(os.path.dirname(__file__),'__tmp_dataset')

		# remove the dataset if it currently exists
		os.system('rm -rf %s' % dataset_dir)

		posts2dataset(dataset_dir,post_file,
					  lambda x: str(x['user_id']),
					  lambda x: map(str,x.get('mentions',[])))

		self.assertTrue(os.path.exists(dataset_dir))
		self.assertTrue(os.path.exists(os.path.join(dataset_dir,'posts.json.gz')))
		self.assertTrue(os.path.exists(os.path.join(dataset_dir,'users.json.gz')))
		self.assertTrue(os.path.exists(os.path.join(dataset_dir,'mention_network.elist')))

		# done
		os.system('rm -rf %s' % dataset_dir)

DEFAULT_TMP_DATASET_DIR = '__tmp_dataset'
def build_tmp_dataset(dataset_dir=DEFAULT_TMP_DATASET_DIR):

	# remove the dataset if it currently exists
	os.system('rm -rf %s' % dataset_dir)

	posts2dataset(dataset_dir,post_file,
				  lambda x: str(x['user_id']),
				  lambda x: map(str,x.get('mentions',[])))

	return

class DatasetBasicFunctionalityTestCase(unittest.TestCase):

	def test_load(self):
		build_tmp_dataset()	

		ds = Dataset(DEFAULT_TMP_DATASET_DIR)

		self.assertEquals(len(list(ds.post_iter())),15)

		os.system('rm -rf %s' % DEFAULT_TMP_DATASET_DIR)

