##
#  Copyright (c) 2015, David Jurgens
#
#  All rights reserved. See LICENSE file for details
##

from geolocate import GIMethod, GIModel

JAKARTA_LAT_LON = (-6.2000, 106.8)

class Jakartr_Model(GIModel):
    '''
    A baseline GIModel that says everyone is in Jakarta, Indonesia, the city
    with the most Twitter users.
    '''

    def __init__(self):
        pass


    def infer_post_location(self, post):
        return JAKARTA_LAT_LON


    def infer_posts_by_user(self, posts):

        # Fill the array with Jakarta's location
        locations = []
        for i in range(0, len(posts)):
            locations.append(JAKARTA_LAT_LON)
        return locations


class Jakartr(GIMethod):
    def __init__(self):
        pass

    def train_model(self, setting, dataset, model_dir):
        # Jakarta is fully unsupervised
        return Jakartr_Model()

    def load_model(self, model_dir, settings):
        # Jakarta is fully unsupervised
        return Jakartr_Model()
