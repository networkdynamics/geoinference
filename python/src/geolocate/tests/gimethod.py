##
#  Copyright (c) 2015, Derek Ruths, Yi Tian Xu, Tyler Finethy, David Jurgens
#
#  All rights reserved. See LICENSE file for details
##

import unittest

import geolocate.gimethod as gim

class GIMethodLoadTestCase(unittest.TestCase):

	def test_load_dummy(self):
		from geolocate.gimethods.dummy.method import DummyMethod

		print 'Num subclasses: %d' % len(gim.gimethod_subclasses())

		self.assertTrue(DummyMethod in gim.gimethod_subclasses())

