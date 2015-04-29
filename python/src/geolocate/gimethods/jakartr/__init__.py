# this is needed for pickle

from method import *
from . import method
import sys
sys.modules['method'] = method 
sys.modules['method.Jakartr_Model'] = method.Jakartr_Model
