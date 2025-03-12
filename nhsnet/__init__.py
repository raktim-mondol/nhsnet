from .models.nhsnet import NHSNet, NHSNetBlock
from .layers.hebbian_conv import HebbianConv2d
from .layers.structured_sparse import StructuredSparseConv2d
from .layers.hh_gating import HodgkinHuxleyGating
from .layers.dynamic_neurogenesis import DynamicNeurogenesisModule
from .utils.pruning import AdaptiveSynapticPruning

__version__ = '0.1.0'

__all__ = [
    'NHSNet',
    'NHSNetBlock',
    'HebbianConv2d',
    'StructuredSparseConv2d',
    'HodgkinHuxleyGating',
    'DynamicNeurogenesisModule',
    'AdaptiveSynapticPruning',
]