from tdsc_abus2023_pytorch import DataSplits
from tdsc_abus2023_pytorch.tdsc import TDSC
from tdsc_abus2023_pytorch.tdsc_tumors import TDSCTumors

def test_tdsc_instantiation():
    """Test basic TDSC dataset instantiation."""
    # Test with download=True
    dataset = TDSC(
        path="./data",
        split=DataSplits.TRAIN,
        download=True
    )
    assert dataset is not None
    assert dataset.path == "./data"
    assert dataset.split == DataSplits.TRAIN
    assert len(dataset.transforms) == 0


def test_tdsc_tumors_instantiation():
    """Test TDSC tumors dataset instantiation."""
    dataset = TDSCTumors(
        path="./data",
        split=DataSplits.TRAIN,
        download=True
    )
    assert dataset is not None
    assert dataset.path == "./data"
    assert dataset.split == DataSplits.TRAIN
    assert len(dataset.transforms) == 0
