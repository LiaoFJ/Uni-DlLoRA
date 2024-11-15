# Official Implementation of Uni-DlLoRA (ACM MM '24)
The official implementation of Uni-DlLoRA: Style Fine-Tuning for Fashion Image Translation (ACM MM '24)

![teaser](./teaser/first.pdf)

## Installation Guide

To use Uni-DlLora, you need to install a compatible version of PyTorch with CUDA support. The version of PyTorch is: torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0.


1. Install a compatible version of PyTorch with CUDA
   ```bash
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
   ```

2. Install the dependencies
   ```bash
    pip install -r requirements.txt
   ```
   
## Inference Guide
Download the checkpoint from 
[Google Drive](https://drive.google.com/uc?export=download&id=1cXpONrwXdyjKqSvN7I9SXPEbWRl2Iznm).
unzip the file, and get the checkpoint.

Generate image using inference.py

```shell
python inference.py --checkpoint_save_dir CHECKPOINT.DIR 
```

## Acknowledgement
This work is supported by Laboratory for Artificial Intelligence in Design (Project Code: RP 3-1) under InnoHK Research Clusters, 
Hong Kong SAR Government.