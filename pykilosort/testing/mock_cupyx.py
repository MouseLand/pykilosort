from mock import Mock
import sys
import types

import scipy

module_name = 'cupyx'
cupyx = types.ModuleType(module_name)
sys.modules[module_name] = cupyx 

cupyx.scipy = Mock(name=module_name+'.scipy')
