# this is needed for pickle

from method import *
from . import method
import sys
sys.modules['method'] = method 
sys.modules['method.Davis_Jr_et_al_Model'] = method.Davis_Jr_et_al_Model
