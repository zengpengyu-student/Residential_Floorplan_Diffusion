## Residential Floor Plan Generation with Diffusion Models
---

## Overview
This repository hosts the implementation of two key research works on residential floor plan generation, focusing on diffusion model-based solutions. It includes the core model from "Residential floor plans: Multi-conditional automatic generation using diffusion models" and the generation module from "Automated residential layout generation and editing using natural language and images". (without outer contour input + 20-scale version)

## Environment Setup

1. Python 3.8+
2. PyTorch 1.10+ (with CUDA support recommended for acceleration)
3. Other dependencies listed in requirements.txt

## Installation
1. Clone this repository:
```bash
git clone https://github.com/zengpengyu-student/Residential_Floorplan_Diffusion.git
cd Residential_Floorplan_Diffusion
```
2. Install dependencies via pip:
```bash
pip install -r requirements.txt
```

## Model Preparation
Pre-trained models are hosted on Google Drive. Follow these steps to set up the models:

1. Download the pre-trained model files from the link below:
[model_stage1.pth](https://drive.google.com/file/d/1ONAu_i2q0FUGJClArBC4mBMwFAchmo1H/view?usp=drive_link)
and [model_stage2.pth](https://drive.google.com/file/d/1cVL4dLMM7j0n3nSKkapjdTWgHD9H45rA/view?usp=drive_link)

2. Create a logs/ directory in the root of the repository (if not exists):
```bash
mkdir -p logs
```

3. Move the downloaded model files (e.g., model_stage1.pth, model_stage2.pth) into the logs/ directory.
Final directory structure should look like:

```plaintext
residential-floorplan-diffusion/ 
├── logs/                          
│   ├── model_stage1.pth          
│   └── model_stage2.pth          
├── predict.py                     
├── requirements.txt               
└── ... (other files)              
```

## Inference

Run the ```predict.py``` script directly to generate residential floor plans. The script will use the pre-trained models in ```logs/``` and output results to a default directory (configurable in the script).

```bash
python predict.py
```

## Customization (Optional)

You can modify key parameters in predict.py to adjust the generation process according to your needs:

1. ```predict_num```: Number of floor plan images to generate in one run (default: ```8```).
2. ```Room_Judgment```: A boolean flag to control room quantity validation. Set to True to enable automatic verification of room count consistency; set to False to skip validation and output directly (default: ```True```).
3. ```path_room```: Path to the input test images (used as conditional inputs for generation; ensure the path points to valid image files).
4. ```save_path_stage1``` and ```save_path_stage2```: Directory to save generated floor plans (default: ```test/```).

## Citation
If you find this repository useful in your research, please consider citing the following papers:

```bash
@article{zeng2024residential,
  title={Residential floor plans: Multi-conditional automatic generation using diffusion models},
  author={Zeng, Pengyu and Gao, Wen and Yin, Jun and Xu, Pengjian and Lu, Shuai},
  journal={Automation in Construction},
  volume={162},
  pages={105374},
  year={2024},
  publisher={Elsevier}
}
```

```bash
@article{zeng2025automated,
  title={Automated residential layout generation and editing using natural language and images},
  author={Zeng, Pengyu and Gao, Wen and Li, Jizhizi and Yin, Jun and Chen, Jiling and Lu, Shuai},
  journal={Automation in Construction},
  volume={174},
  pages={106133},
  year={2025},
  publisher={Elsevier}
}
```
