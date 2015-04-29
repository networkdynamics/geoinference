# this is needed for pickle

from method import *
from . import method
import sys
sys.modules['method'] = method 
sys.modules['method.SpatialLabelPropagationModel'] = method.SpatialLabelPropagationModel
