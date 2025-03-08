from enum import Enum

class DataSplits(str, Enum):
    TRAIN = "Train"
    VALIDATION = "Validation"
    TEST = "Test"

    def __str__(self):
        return self.value 