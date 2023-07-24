# DDPM (Denoising Diffusion Probabilistic Models)
- The process of Diffusion Model
<img src = "https://github.com/Sangh0/Generative/blob/main/DDPM/figures/diffusion_process.png?raw=true">
- [paper](https://arxiv.org/abs/2006.11239)

### Training
```
python3 main.py --dataset mnist --store_path ./ddpm_best.pt --epochs 30
```

### MNIST handwritten Examples
- The generated examples for 1 epoch in training
<img src = "https://github.com/Sangh0/Generative/blob/main/DDPM/figures/1epoch.png?raw=true" width=300>

- The generated examples for 20 epoch in training
<img src = "https://github.com/Sangh0/Generative/blob/main/DDPM/figures/20epoch.png?raw=true" width=300>