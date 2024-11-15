# Official Implementation of Uni-DlLoRA (ACM MM '24)
The official implementation of [Uni-DlLoRA: Style Fine-Tuning for Fashion Image Translation](https://dl.acm.org/doi/pdf/10.1145/3664647.3681459) (ACM MM '24)
![teaser](./teaser/teaser.png?raw=true)

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

## Citation
```bibtex
@inproceedings{10.1145/3664647.3681459,
author = {Liao, Fangjian and Zou, Xingxing and Wong, Waikeung},
title = {Uni-DlLoRA: Style Fine-Tuning for Fashion Image Translation},
year = {2024},
isbn = {9798400706868},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3664647.3681459},
doi = {10.1145/3664647.3681459},
abstract = {Image-to-image (i2i) translation has achieved notable success, yet remains challenging in scenarios like real-to-illustrative style transfer of fashion. Existing methods focus on enhancing the generative model with diversity while lacking ID-preserved domain translation. This paper introduces a novel model named Uni-DlLoRA to release this constraint. The proposed model combines the original images within a pretrained diffusion-based model using the proposed Uni-adapter extractors, while adopting the proposed Dual-LoRA module to provide distinct style guidance. This approach optimizes generative capabilities and reduces the number of additional parameters required. In addition, a new multimodal dataset featuring higher-quality images with captions built upon an existing real-to-illustration dataset is proposed. Experimentation validates the effectiveness of our proposed method.},
booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
pages = {6404–6413},
numpages = {10},
keywords = {denoising diffusion probabilistic models, fashion synthesis, image-to-image translation},
location = {Melbourne VIC, Australia},
series = {MM '24}
}
```

## Acknowledgement
This work is supported by Laboratory for Artificial Intelligence in Design (Project Code: RP 3-1) under InnoHK Research Clusters, 
Hong Kong SAR Government.