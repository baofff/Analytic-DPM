# Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models

Code for the paper [Analytic-DPM: an Analytic Estimate of the Optimal Reverse Variance in Diffusion Probabilistic Models
](https://arxiv.org/abs/2201.06503)

## Requirements
pytorch=1.9.0

## Run experiments

You can change the `phase` variable in the code to determine the specific experiment you run. 

For example, setting `phase = "sample_analytic_ddpm"` will run sampling using the Analytic-DDPM.

You can find all available phases in `run_xxx.py`.


### CIFAR10
```
$ cd cifar_imagenet_codes
$ python run_cifar10.py
```

### CelebA 64x64
```
$ cd celeba_lsun_codes
$ python run_celeba.py
```

### Imagenet 64x64
```
$ cd cifar_imagenet_codes
$ python run_imagenet64.py
```

### LSUN Bedroom
```
$ cd celeba_lsun_codes
$ python run_lsun_bedroom.py
```

## Pretrained models and precalculated statistics

* CIFAR10 model: [[checkpoint](https://drive.google.com/file/d/1WyoUFDQeJUJblAT85Tc1ntbqklvqMP3J/view?usp=sharing)] trained by ourselves

* CelebA 64x64 model: [[checkpoint](https://drive.google.com/file/d/1R_H-fJYXSH79wfSKs9D-fuKQVan5L-GR/view?usp=sharing)] from https://github.com/ermongroup/ddim

* Imagenet 64x64 model: [[checkpoint](https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_uncond_100M_1500K.pt)] from https://github.com/openai/improved-diffusion

* LSUN Bedroom model: [[checkpoint](https://heibox.uni-heidelberg.de/d/01207c3f6b8441779abf/)] from https://github.com/pesser/pytorch_diffusion

* Precalculated Gamma vectors: [[link](https://drive.google.com/file/d/1pnwxNFY-0P_IZaTVP1zNBxzKb3T1QeD7/view?usp=sharing)]
