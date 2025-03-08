# TDSC-ABUS2023 PyTorch Dataset

A PyTorch-compatible dataset package containing volumetric data from the TDSC-ABUS2023 collection (Tumor Detection, Segmentation and Classification Challenge on Automated 3D Breast Ultrasound).

## Dataset Description

The dataset contains 200 3D volumes with refined tumor labels, collected using an Automated 3D Breast Ultrasound (ABUS) system (Invenia ABUS, GE Healthcare) at Harbin Medical University Cancer Hospital, Harbin, China. All annotations were created and verified by an experienced radiologist.

### Technical Specifications
- **Image Dimensions**: Varying between 843×546×270 and 865×682×354
- **Pixel Spacing**: 
  - X-Y plane: 0.200 mm × 0.073 mm
  - Z-axis (between slices): ~0.475674 mm
- **File Format**: .nrrd
- **Annotations**: Voxel-level segmentation
  - 0: Background
  - 1: Tumor

### Dataset Split
The dataset is stratified sampled from all 200 cases and divided into:

- **Training Set**: 100 cases
  - Used for training robust models
- **Validation Set**: 30 cases
  - Open validation set for algorithm verification
  - Sized to prevent test set distribution leakage
- **Test Set**: 70 cases
  - Closed set for final leaderboard evaluation
  - Ensures fair comparison between methods

## Installation

You can install this package via pip:

```bash
pip install tdsc-abus2023-pytorch
```

## Usage

```python
from tdsc-abus2023-pytorch import TDSC, DataSplits

# Initialize dataset with automatic download
dataset = TDSC(
    path="./data",
    split=DataSplits.TRAIN,
    download=True
)

# Access a sample
volume, mask, label = dataset[0]
```

## Data Structure
```
data/
└── tdsc/
    ├── Train/
    │   ├── DATA/
    │   └── MASK/
    ├── Validation/
    │   ├── DATA/
    │   └── MASK/
    └── Test/
        ├── DATA/
        └── MASK/
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{luo2025tumordetectionsegmentationclassification,
    title={Tumor Detection, Segmentation and Classification Challenge on Automated 3D Breast Ultrasound: The TDSC-ABUS Challenge}, 
    author={Gongning Luo and Mingwang Xu and Hongyu Chen and Xinjie Liang and Xing Tao and Dong Ni and Hyunsu Jeong and Chulhong Kim and Raphael Stock and Michael Baumgartner and Yannick Kirchhoff and Maximilian Rokuss and Klaus Maier-Hein and Zhikai Yang and Tianyu Fan and Nicolas Boutry and Dmitry Tereshchenko and Arthur Moine and Maximilien Charmetant and Jan Sauer and Hao Du and Xiang-Hui Bai and Vipul Pai Raikar and Ricardo Montoya-del-Angel and Robert Marti and Miguel Luna and Dongmin Lee and Abdul Qayyum and Moona Mazher and Qihui Guo and Changyan Wang and Navchetan Awasthi and Qiaochu Zhao and Wei Wang and Kuanquan Wang and Qiucheng Wang and Suyu Dong},
    year={2025},
    eprint={2501.15588},
    archivePrefix={arXiv},
    primaryClass={eess.IV},
    url={https://arxiv.org/abs/2501.15588}, 
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
