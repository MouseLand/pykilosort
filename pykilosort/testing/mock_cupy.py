from mock import Mock
import sys
import types

import numpy

module_name = 'cupy'
cupy = numpy
sys.modules[module_name] = cupy
cupy.asnumpy = lambda x: x # noop
