from enum import Enum

class DataSplits(str, Enum):
    TRAIN = "Train"
    VALIDATION = "Validation"
    TEST = "Test"

    def __str__(self):
        return self.value

# Remove or fix the problematic import
# from tdsc_tumors import TDSCTumors  # This was causing the error

__version__ = "0.1.0"