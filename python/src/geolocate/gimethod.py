##
#  Copyright (c) 2015, Derek Ruths, Yi Tian Xu, Tyler Finethy, David Jurgens
#
#  All rights reserved. See LICENSE file for details
##

import os, os.path
from importlib import import_module
import abc
import glob

import zen

import logging

logger = logging.getLogger(os.path.basename(__file__))

__subclass_import_completed = False

def __import_gimethod_subclasses():
	import gimethods

	dirpath = os.path.dirname(gimethods.__file__)
	modules = []

	logger.debug('searching for modules in %s' % dirpath)
	for d in filter(lambda x: os.path.isdir(os.path.join(dirpath,x)),os.listdir(dirpath)):
		path = os.path.join(dirpath,d)
		files = os.listdir(path)
		files = filter(lambda x: x == 'method.py', files)
		mods = map(lambda x: x.replace('.py',''), files)
		modules += ['geolocate.gimethods.%s.%s' % (d,m) for m in mods]

	# load all the modules
	logger.debug('loading all modules found')
	for m in modules:
		import_module('%s' % m)

	# done

def gimethod_subclasses():
	global __subclass_import_completed

	if not __subclass_import_completed:
		logger.debug('building classes')
		__import_gimethod_subclasses()
		__subclass_import_completed = True

	return GIMethod.__subclasses__() 

class GIMethod(object):
	"""
	This is the abstract base class for all geographical inference systems supported
	by the geoinference framework.  All subclasses must explicitly subclass this 
	class and implement the `GIMethod.train_model(...)` and 
	`GIMethod.load_model(...)`  methods.
	"""
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def train_model(self,settings,dataset,model_dir):
		"""
		This method build a geoinference model using the settings and dataset
		provided.  The model should be stored in the directory `model_dir` which
		is guaranteed to exist.

		*Returns*:
		A subclass of `GIModel` specific to the subclass of `GIMethod` that has
		implemented this method.
		"""
		pass

	@abc.abstractmethod
	def load_model(self,model_dir,settings):
		"""
		This method loads a subclass of `GIModel` from the directory `model_dir`
		which, presumably, was constructed by a call to `GIMethod.train_model(...)`.

		*Returns*:
		A subclass of `GIModel` specific to the subclass of `GIMethod` that has
		implemented this method.
		"""
		pass

class GIModel(object):
	"""
	This is the abstract base class for all geographical inference system models.
	Models are first produced through a training procedure.  After training has
	been completed, the model can be used to infer locations.

	When subclassed, the `infer_post_location(...)` method must be implemented and 
	the `infer_posts_by_user(...)` may optionally be implemented.
	"""

	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def infer_post_location(self,post):
		"""
		This method infers the location for the post specified.

		*Returns*:
		A lat-long tuple.
		"""
		return

	def infer_posts_locations_by_user(self,user_id,posts):
		"""
		This method infers the locations for each of the posts in the
		list provided.  Each post is guaranteed to belong to the same user.

		This method can be overridden by the implementing class. By default,
		it simply calls `GIModel.infer_post_location(...)` for each post in 
		the list.

		*Returns*:
		A list of lat-long tuples, R, such that R[i] is the lat-long for the
		ith post passed in.
		"""
		return [self.infer_post_location(post) for post in posts]		
