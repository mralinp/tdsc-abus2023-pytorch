from .enums import DataSplits
from .tdsc import TDSC
from .tdsc_tumors import TDSCTumors

__all__ = ['TDSC', 'TDSCTumors', 'DataSplits']

# Remove or fix the problematic import
# from tdsc_tumors import TDSCTumors  # This was causing the error

__version__ = "0.1.0"