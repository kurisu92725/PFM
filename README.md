# Perturbed Flow Matching for Structure-Based Drug Design

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

This is the official implementation of the paper "Perturbed Flow Matching for Structure-Based Drug Design". [[PDF]]

<p align="center">
  <img src="PFM_mf.png"/> 
</p>

## Installation

### Key Dependencies
- Python | 3.8.18
- PyTorch | 2.0.1+cu117
- PyTorch Geometric | 2.4.0
- RDKit | 2022.03.5
- AutoDock Vina | 1.2.0

For all dependencies, please refer to `requirements.txt`.

### AutoDock Vina Setup
1. Install AutoDock Vina
```bash
pip install meeko==0.1.dev3 scipy pdb2pqr vina==1.2.2 
python -m pip install git+https://github.com/Valdes-Tresanco-MS/AutoDockTools_py3
```

2. Replace `vina.py` file in the package with `utils/vina.py`

## Data

### Preprocessed Data
The preprocessed data can be found [here](https://drive.google.com/drive/folders/1j21cc7-97TedKh_El5E34yI8o5ckI7eK).

### Data Processing
If you want to process your own data, please refer to [TargetDiff](https://github.com/guanjq/targetdiff).

## Training
```bash
python scripts/train_fm.py configs/training_x_predictor.yml
python scripts/train_fm.py configs/training_h_predictor.yml
python scripts/train_fm.py configs/training_pfm.yml
python scripts/train_fm.py configs/training_pfm_g.yml
```

Update the dataset paths in the selected configuration before training. PFM
and PFM-G load `model_params/x_predictor.pth` and
`model_params/h_predictor.pth` by default.

## Sampling
```bash
python scripts/sample_fm.py configs/sampling_pfm.yml --result_path outputs/pfm
python scripts/sample_fm.py configs/sampling_pfm_g.yml --result_path outputs/pfm_g
```

PFM and PFM-G use different checkpoint structures. 

## Evaluation
```bash
python scripts/evaluate_fm.py outputs/pfm --docking_mode vina_score \
  --protein_root /path/to/crossdock_2020
```
Please note that {PROTEIN_ROOT} is the original dataset CrossDocked2020 v1.1, which can be downloaded [here](https://bits.csb.pitt.edu/files/crossdock2020/).

## Citation

If you find this code useful for your research, please cite our paper:
```
[...]
```

### Acknowledgments

This project builds upon several previous works. We recommend citing the following papers:
```
@inproceedings{guan3d,
  title={3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction},
  author={Guan, Jiaqi and Qian, Wesley Wei and Peng, Xingang and Su, Yufeng and Peng, Jian and Ma, Jianzhu},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
```
@article{tong2023simulation,
  title={Simulation-free schr$\backslash$" odinger bridges via score and flow matching},
  author={Tong, Alexander and Malkin, Nikolay and Fatras, Kilian and Atanackovic, Lazar and Zhang, Yanlei and Huguet, Guillaume and Wolf, Guy and Bengio, Yoshua},
  journal={arXiv preprint arXiv:2307.03672},
  year={2023}
}
```
