# AdaptSplat: Adapting Vision Foundation Models for Feed-Forward 3D Gaussian Splatting

<p align="center">
  <a href="https://arxiv.org/abs/2605.10239">
    <img src="https://img.shields.io/badge/arXiv-2605.10239-b31b1b.svg" alt="arXiv">
  </a> &nbsp;
  <a href="#">
    <img src="https://img.shields.io/badge/Project%20Page-4CAF50?style=flat&logo=googlechrome&logoColor=white" alt="Project Page">
  </a> &nbsp;
  <a href="https://huggingface.co/koun123/AdaptSplat/tree/main">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-FFD21E?style=flat&labelColor=FFD21E&color=FFD21E" alt="HuggingFace">
  </a>
</p>

This repository contains the official open-source implementation of "AdaptSplat: Adapting Vision Foundation Models for Feed-Forward 3D Gaussian Splatting". If you find this project helpful, please consider giving us a star on GitHub ⭐️✨

---

## TODO

- [x] ~~Release DL3DV model weights (960×540)~~
- [x] ~~Release inference code~~
- [ ] Release Stage 1 & Stage 2 training code
- [ ] Release Stage 3 training code

---

## Environment Setup

### System Requirements

- Python 3.10
- CUDA 12.1 (tested; other CUDA 12.x versions may work)
- GCC ≥ 7

### Step-by-Step Installation

**Step 1: Create a conda environment**

```bash
conda create -n adaptsplat python=3.10 -y
conda activate adaptsplat
```

**Step 2: Install PyTorch with CUDA 12.1**

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

**Step 3: Install all dependencies**

```bash
pip install -r requirements.txt
```

---

## Model Weights

### Pretrained Backbone Weights (DINOv3-distilled ConvNeXt)

Download the pretrained backbone weights and place them in the `pretrian_weight/` directory:

| Model | Size | Download |
|---|---|---|
| ConvNeXt-Base | 338 MB | [DINOv3 Downloads](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/) |

> **Note:** Access to DINOv3 weights requires submitting a request to the DINOv3 team via the official download page.

```
pretrian_weight/
└── dinov3_convnext_base_pretrain_lvd1689m-801f2ba9.pth
```

### AdaptSplat Checkpoint

| Checkpoint | Training Data | Resolution | Download |
|---|---|---|---|
| `ckpt_adaptsplat_dl3dv_960.pt`  | DL3DV | 960×540 | [HuggingFace](https://huggingface.co/koun123/AdaptSplat/tree/main) |

Place the checkpoint at `checkpoints/ckpt_adaptsplat_dl3dv_960.pt`.

---

## Data Preparation

### DL3DV (140 benchmark scenes)

**Original Dataset Download**

Download the DL3DV benchmark dataset from [HuggingFace](https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark/tree/main) and place it under `data_dl3dv/`.

**Data Preprocessing**

Run the provided preprocessing script to perform undistortion and camera format conversion:

```bash
python data/process_dl3dv.py
```

> The preprocessing script is adapted from [LongLRM](https://github.com/arthurhero/Long-LRM).

**Expected directory structure after preprocessing:**

```
data_dl3dv/
├── dl3dv_bechmark_140_hf.txt      # scene list (one path per line, e.g. dl3dv_benchmark/<hash>/opencv_cameras.json)
└── dl3dv_benchmark/
    └── <scene_hash>/
        ├── opencv_cameras.json    # per-frame intrinsics + w2c in OpenCV convention
        └── images_undistort/
            ├── frame_00001.png
            └── ...
```

The k-means input frame indices (pre-computed) are provided at `data/dl3dv_fold_8_kmeans_input_idx.json`.

---

## Inference

### Single-GPU

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --config configs/inference.yaml
```

### Multi-GPU (DDP)

Scenes are distributed across GPUs; each GPU runs inference independently and Rank 0 aggregates metrics at the end.

```bash
# 8-GPU inference on DL3DV
torchrun --nproc_per_node=8 inference_ddp.py --config configs/inference.yaml
```


## Acknowledgement

This project is built upon [Long-LRM](https://github.com/arthurhero/Long-LRM) and [MVP](https://github.com/Gynjn/MVP). We sincerely thank the authors and contributors of these excellent open-source works.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{adaptsplat2026,
  title={AdaptSplat: Adapting Vision Foundation Models for Feed-Forward 3D Gaussian Splatting},
  author={Mingwei Xing, Xinliang Wang, Yifeng Shi},
  journal={arXiv preprint arXiv:2605.10239},
  year={2026}
}
```
