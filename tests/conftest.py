import sys
import types

import numpy as np
import nrrd
import pandas as pd
import pytest


def _install_stub_module(name):
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


try:
    import torch  # noqa: F401
except ImportError:
    torch_stub = _install_stub_module("torch")
    utils_stub = _install_stub_module("torch.utils")
    data_stub = _install_stub_module("torch.utils.data")

    class Dataset:
        pass

    data_stub.Dataset = Dataset
    utils_stub.data = data_stub
    torch_stub.utils = utils_stub


try:
    import gdown  # noqa: F401
except ImportError:
    gdown_stub = _install_stub_module("gdown")

    def _unavailable_download(*args, **kwargs):
        raise RuntimeError("gdown is not installed; mock gdown.download in this test")

    gdown_stub.download = _unavailable_download


VOLUME_SHAPE = (6, 5, 4)  # (z, y, x)


@pytest.fixture
def volume_data():
    volume = np.arange(np.prod(VOLUME_SHAPE), dtype=np.int32).reshape(VOLUME_SHAPE)
    mask = volume + 1000
    return volume, mask


@pytest.fixture
def dataset_dir(tmp_path, volume_data):
    """Build a minimal, synthetic TDSC-style dataset on disk (no network access)."""
    volume, mask = volume_data

    split_dir = tmp_path / "Train"
    data_dir = split_dir / "DATA"
    mask_dir = split_dir / "MASK"
    data_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    nrrd.write(str(data_dir / "case1.nrrd"), volume)
    nrrd.write(str(mask_dir / "case1.nrrd"), mask)

    pd.DataFrame({
        "Case_id": [1],
        "Label": ["M"],
        "Data_path": ["DATA/case1.nrrd"],
        "Mask_path": ["MASK/case1.nrrd"],
    }).to_csv(split_dir / "labels.csv", index=False)

    # Bounding box: z 2:4, y 1:3, x 1:3 (within the 6x5x4 volume above)
    pd.DataFrame({
        "id": [1],
        "c_x": [2.0], "c_y": [2.0], "c_z": [3.0],
        "len_x": [2.0], "len_y": [2.0], "len_z": [2.0],
    }).to_csv(split_dir / "bbx_labels.csv", index=False)

    return str(tmp_path)
