# TDSC-ABUS2023 PyTorch Dataset

A PyTorch-compatible dataset package containing volumetric data from the TDSC-ABUS2023 collection.

## Description

This package provides easy access to the TDSC-ABUS2023 dataset, which consists of volumetric medical imaging data. The dataset is split into:
- 100 training volumes
- 30 validation volumes
- 70 test volumes

## Installation

You can install this package via pip:

```bash
pip install tdsc-abus2023-pytorch
```

## Usage

```python
from tdsc_abus2023 import dataloader

dataset = TDSCABUS2023Dataset(root_dir='path/to/dataset')
```

## Dataset Structure

The dataset contains volumetric data organized as follows:
- Training set: 100 volumes
- Validation set: 30 volumes
- Test set: 70 volumes

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
