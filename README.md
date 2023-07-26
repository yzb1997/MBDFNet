# MBDFNet: Multi-scale Bidirectional Dynamic Feature Fusion Network for Efficient Image Deblurring

This repository provides the official PyTorch implementation of the following paper:

> MBDFNet: Multi-scale Bidirectional Dynamic Feature Fusion Network for Efficient Image Deblurring
>
> Zhongbao Yang, Jinshan Pan
>
> In ICME 2023. 
>
>
> Abstract: Existing deep image deblurring models achieve favorable results with growing model complexity. However, these models cannot be applied to those low-power devices with resource constraints (e.g., smart phones) as these models usually have lots of network parameters and require computational costs. To overcome this problem, we develop a multi-scale bidirectional dynamic feature fusion network (MBDFNet), a lightweight deep deblurring model, for efficient image deblurring. The proposed MBDFNet progressively restores multi-scale latent clear images from blurry input based on a multi-scale framework. To better utilize the features from coarse scales, we propose a bidirectional gated dynamic fusion module so that the most useful information of the features from coarse scales are kept to facilitate the estimations in the finer scales. We solve the proposed MBDFNet in an end-to-end manner and show that it has fewer network parameters and lower FLOPs values, where the FLOPs value of the proposed MBDFNet is at least $6\times$ smaller than the state-of-the-art methods.Both quantitative and qualitative evaluations show that the proposed MBDFNet achieves favorable performance in terms of model complexity while having competitive performance in terms of accuracy against state-of-the-art methods. 

---

## Dependencies

- pytorch 1.11
- Cuda 11.3
- python 3.8

## Get Started

### Create anaconda environment 

```python
conda create -n MBDFNet python==3.8
```

### Install dependencies

```python
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

cd ./MBDFNet
pip install -r requirements.txt
pip install thop
pip install einops
pip install mmcv
pip install cupy_cuda111
```

### Compile wavelet

```python
cd ./basicsr/pytorch_wavelets
pip install .
```

### Compile code

```python
cd ../../
python setup.py develop
```

## Test

### Pretrained model path

```python
./experiments/model.pth
```

### Generate image by MBDFNet
```python
cd ./inferenceCode
python main.py
        # You can change the following code for inference
        # Line 32: your datasets path
        # Line 49: MBDFNet model path
        # Line 54: save path (the path of generate image by MBDFNet)
``` 

### Calculate the psnr, ssim of the generated image

```python
cd ../
python test_PSNR.py
        # You can change the following code for test
        # Line 25: gt path
        # Line 85: model name
        # Line 86: the path of generate image by MBDFNet
```

## Acknowledgment: 
This code is based on the [BasicSR](https://github.com/XPixelGroup/BasicSR)
