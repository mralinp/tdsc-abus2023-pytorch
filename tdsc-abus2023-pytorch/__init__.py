import os
from enum import Enum
from tdsc_tumors import TDSCTumors
from tdsc import TDSC

class DataSplits(str, Enum):
    TRAIN = "Train"
    VALIDATION = "Validation"
    TEST = "Test"

    def __str__(self):
        return self.value

__version__ = "0.1.0"