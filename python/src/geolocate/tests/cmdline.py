##
#  Copyright (c) 2015, Derek Ruths
#
#  All rights reserved. See LICENSE file for details
##

import unittest
import geolocate.app as app
import dataset
import os, os.path
import logging

from geolocate.gimethods.dummy.method import DummyMethod, DummyModel

logger = logging.getLogger(os.path.dirname(__file__))

class CommandLineTestCase(unittest.TestCase):

	def test_build_dataset(self):
		dirname = os.path.dirname(__file__)
		dataset_dir = os.path.join(dirname,'__tmp_dataset')
		posts_file = os.path.join(dirname,'posts.json.gz')

		app.build_dataset([dataset_dir,posts_file,'user_id','mentions'])

		# check if it's there
		ds = dataset.Dataset(dataset_dir)
		self.assertEquals(len(list(ds.post_iter())),15)

		os.system('rm -rf %s' % dataset_dir)

	def test_train_dummy(self):
		DummyMethod.clear()

		# make a dataset
		dirname = os.path.dirname(__file__)
		dataset_dir = os.path.join(dirname,'__tmp_dataset')
		posts_file = os.path.join(dirname,'posts.json.gz')

		app.build_dataset([dataset_dir,posts_file,'user_id','mentions'])

		PHONY_SETTINGS_FILE = os.path.join(dirname,'__tmp_settings')
		PHONY_MODEL_DIR = 'foobar' # this doesn't have to exist - Dummy won't write to it

		fh = open(PHONY_SETTINGS_FILE,'w')
		fh.write('{}')
		fh.close()

		# run the trainer
		logger.debug('running training...')
		args = ['DummyMethod',PHONY_SETTINGS_FILE,dataset_dir,PHONY_MODEL_DIR]
		logger.debug('args = %s' % str(args))
		app.train(args)

		# check and see if train was called
		self.assertTrue(DummyMethod.train_called)

		# destroy the dataset
		os.system('rm -rf %s' % dataset_dir)

		# delete the settings file
		os.system('rm %s' % PHONY_SETTINGS_FILE)

	def test_infer_posts_dummy(self):
		DummyMethod.clear()
		DummyModel.clear()

		# make a dataset
		dirname = os.path.dirname(__file__)
		dataset_dir = os.path.join(dirname,'__tmp_dataset')
		posts_file = os.path.join(dirname,'posts.json.gz')

		app.build_dataset([dataset_dir,posts_file,'user_id','mentions'])

		PHONY_MODEL_DIR = 'foobar' # this doesn't have to exist - Dummy won't load it
		INFER_FILE = os.path.join(dirname,'__tmp_infer')

		# run the infer
		logger.debug('running inference...')
		args = ['-f','DummyMethod',PHONY_MODEL_DIR,dataset_dir,INFER_FILE]
		logger.debug('args = %s' % str(args))
		app.infer(args,False)

		# check the contents of the infer file
		fh = open(INFER_FILE,'r')
		num_posts = 0
		fhiter = iter(fh)
		fhiter.next() # skip the header

		for line in fhiter:
			data = line.split()
			num_posts += 1

			self.assertEquals(-1,float(data[1]))
			self.assertEquals(-1,float(data[2]))

		self.assertEquals(num_posts,DummyModel.num_posts_inferred)

		# destroy the dataset
		os.system('rm -rf %s' % dataset_dir)

		# destroy the output file
		os.system('rm %s' % INFER_FILE)

	
	def test_infer_users_dummy(self):
		DummyMethod.clear()
		DummyModel.clear()

		# make a dataset
		dirname = os.path.dirname(__file__)
		dataset_dir = os.path.join(dirname,'__tmp_dataset')
		posts_file = os.path.join(dirname,'posts.json.gz')

		app.build_dataset([dataset_dir,posts_file,'user_id','mentions'])

		PHONY_MODEL_DIR = 'foobar' # this doesn't have to exist - Dummy won't load it
		INFER_FILE = os.path.join(dirname,'__tmp_infer')

		# run the infer
		logger.debug('running inference...')
		args = ['-f','DummyMethod',PHONY_MODEL_DIR,dataset_dir,INFER_FILE]
		logger.debug('args = %s' % str(args))
		app.infer(args,True)

		# check the contents of the infer file
		fh = open(INFER_FILE,'r')
		num_posts = 0
		fhiter = iter(fh)
		fhiter.next() # skip the header

		for line in fhiter:
			data = line.split()
			num_posts += 1

			self.assertEquals(-1,float(data[1]))
			self.assertEquals(-1,float(data[2]))

		self.assertEquals(num_posts,DummyModel.num_posts_inferred)

		# destroy the dataset
		os.system('rm -rf %s' % dataset_dir)

		# destroy the output file
		os.system('rm %s' % INFER_FILE)

