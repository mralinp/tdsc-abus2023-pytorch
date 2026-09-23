# TDSC-ABUS2023 PyTorch Dataset

[![PyPI version](https://img.shields.io/pypi/v/tdsc-abus2023-pytorch)](https://pypi.org/project/tdsc-abus2023-pytorch/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/mralinp/tdsc-abus2023-pytorch/python-publish.yml?branch=main)](https://github.com/mralinp/tdsc-abus2023-pytorch/actions)

A lightweight PyTorch `Dataset` for the **TDSC-ABUS2023** challenge (Tumor Detection, Segmentation, and Classification on Automated 3D Breast Ultrasound). It downloads the data on first use and hands back ready-to-train volumes, masks, labels, and tumor bounding boxes — no manual file wrangling required.

```python
from tdsc_abus2023_pytorch import TDSC, DataSplits

dataset = TDSC(path="./data", split=DataSplits.TRAIN, download=True)
volume, mask, label, bbox = dataset[0]
```

![Sample case: full axial slice with tumor bounding box and mask overlay, next to the cropped tumor region returned by TDSCTumors](https://raw.githubusercontent.com/mralinp/tdsc-abus2023-pytorch/main/assets/sample_case.png)

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [API Reference](#api-reference)
- [On-Disk Layout](#on-disk-layout)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

## Installation

```bash
pip install tdsc-abus2023-pytorch
```

Requires **Python 3.9+**. The only runtime dependencies are `torch`, `numpy`, `pandas`, `pynrrd`, and `gdown` — nothing else is pulled in.

## Quick Start

### Full volumes

```python
from tdsc_abus2023_pytorch import TDSC, DataSplits

dataset = TDSC(path="./data", split=DataSplits.TRAIN, download=True)
volume, mask, label, bbox = dataset[0]
# volume, mask : np.ndarray            — full 3D ultrasound volume / segmentation mask
# label        : int                   — 0 = Malignant, 1 = Benign
# bbox         : ((x0, y0, z0), (x1, y1, z1)) — tumor bounding box, in the volume's native coordinates
```

### Tumor crops only

`TDSCTumors` returns the volume and mask already cropped to the tumor's bounding box — handy for classification or patch-based segmentation.

```python
from tdsc_abus2023_pytorch import TDSCTumors, DataSplits

dataset = TDSCTumors(path="./data", split=DataSplits.TRAIN, download=True)
volume, mask, label = dataset[0]
```

### Changing the anatomical view

`ViewTransformer` transposes each volume/mask pair into a different anatomical plane before it's returned.

```python
from tdsc_abus2023_pytorch import TDSC, DataSplits, ViewTransformer, ViewTransposeConfig

transformer = ViewTransformer(view=ViewTransposeConfig.CORONAL)
dataset = TDSC(path="./data", split=DataSplits.TRAIN, transforms=[transformer])
volume, mask, label, bbox = dataset[0]
```

![Axial, coronal, and sagittal views of the same tumor produced by ViewTransformer](https://raw.githubusercontent.com/mralinp/tdsc-abus2023-pytorch/main/assets/views.png)

### Custom transforms

Any callable of the form `(volume, mask) -> (volume, mask)` can be used as a transform. Pass several to `transforms=[...]` and they run in order.

```python
class MyTransform:
    def __call__(self, volume, mask):
        return volume, mask  # your logic here

dataset = TDSC(path="./data", split=DataSplits.TRAIN, transforms=[ViewTransformer(ViewTransposeConfig.AXIAL), MyTransform()])
```

## Dataset

200 3D breast ultrasound volumes acquired with an **Invenia ABUS (GE Healthcare)** system at Harbin Medical University Cancer Hospital, China. Tumor segmentation masks and bounding boxes were created and verified by experienced radiologists.

| Split          | Cases | Malignant | Benign |
| -------------- | ----- | --------- | ------ |
| Train          | 100   | 58        | 42     |
| Validation     | 30    | 17        | 13     |
| Test           | 70    | 40        | 30     |

- **Volume size**: varies between 843×546×270 and 865×682×354 voxels
- **Voxel spacing**: 0.200 mm × 0.073 mm (X–Y) × ~0.475674 mm (Z)
- **File format**: `.nrrd`
- **Mask labels**: `0` background, `1` tumor

On first use (`download=True`), each requested split is fetched from Google Drive, extracted, and cached at `path/<Split>/`; subsequent runs reuse the cached copy without touching the network.

## API Reference

| Class                             | Returns                                    | Notes                                   |
| ---------------------------------- | ------------------------------------------- | ---------------------------------------- |
| `TDSC(path, split, transforms, download)` | `(volume, mask, label, bbox)`       | Full volume and mask                    |
| `TDSCTumors(path, split, transforms, download)` | `(volume, mask, label)`       | Cropped to the tumor bounding box        |
| `ViewTransformer(view)`            | `(volume, mask)`                            | `view` is `ViewTransposeConfig.{AXIAL, CORONAL, SAGITTAL}` |
| `DataSplits`                       | —                                            | `DataSplits.{TRAIN, VALIDATION, TEST}`   |

All datasets accept a plain string instead of the enum (e.g. `split="Train"`).

## On-Disk Layout

```
data/
├── Train/
│   ├── DATA/
│   ├── MASK/
│   ├── labels.csv
│   └── bbx_labels.csv
├── Validation/
│   └── ...
└── Test/
    └── ...
```

## Development

```bash
git clone https://github.com/mralinp/tdsc-abus2023-pytorch.git
cd tdsc-abus2023-pytorch
pip install -r requirements.txt
pip install pytest pytest-cov
pytest
```

The test suite runs entirely offline against a small synthetic dataset — it never downloads the real data.

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{luo2025tumordetectionsegmentationclassification,
    title={Tumor Detection, Segmentation and Classification Challenge on Automated 3D Breast Ultrasound: The TDSC-ABUS Challenge},
    author={Gongning Luo and others},
    year={2025},
    eprint={2501.15588},
    archivePrefix={arXiv},
    primaryClass={eess.IV},
    url={https://arxiv.org/abs/2501.15588},
}
```

## License

Released under the [MIT License](LICENSE).

Contributions are welcome — fork the repository, make your changes, and open a pull request.
