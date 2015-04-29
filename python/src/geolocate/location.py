##
#  Copyright (c) 2015, Derek Ruths
#
#  All rights reserved. See LICENSE file for details
##

def exact_location(latitude,longitude,name=None):
	"""
	A helper function for creating a location object for an exact position on the Earth.
	"""
	return Location(center_lat=latitude,
					center_long=longitude,
					surface_area=0,
					bounding_radius=0,
					name=name)

class Location(object):
	"""
	Instances of this class represent a location (either exact or a region) on Earth.
	"""

	def __init__(self,**kwargs):
		self._center_lat = kwargs.pop('center_lat')
		self._center_long = kwargs.pop('center_long')
		self._surface_area = kwargs.pop('surface_area')
		self._bounding_radius = kwargs.pop('bounding_radius')
		
		self._name = kwargs.pop('name',None)
		self._city = kwargs.pop('city',None)
		self._state_province = kwargs.pop('city_province',None)
		self._country = kwargs.pop('country',None)
		self._continent = kwargs.pop('continent',None)

	@property
	def center_lat(self):
		return self._center_lat

	@property
	def center_long(self):
		return self._center_long

	@property
	def center(self):
		return (self._center_lat,self._center_long)

	@property
	def surface_area(self):
		return self._surface_area

	@property
	def bounding_radius(self):
		return self._bounding_radius

	@property
	def name(self):
		return self._name

	def __contains__(self,location):
		# TODO: test to see if the location given is inside this location
		raise NotImplementedError
		
	@property
	def city(self):
		return self._city
		
	@property
	def state_province(self):
		return self._state_province

	@property
	def country(self):
		return self._country

	@property
	def continent(self):
		return self._continent
