import torch
import torchvision
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def to_pil(x):
    # x shape [C,H,W] to numpy [H,W,C]
    x = x.permute(1, 2, 0)
    x = (x.detach().cpu().numpy() * 255).astype(np.uint8)
    x = Image.fromarray(x).convert("RGB") 
    return x
    
def show_img(img):
    # img_t [C,H,W]
    plt.figure(figsize=(6,6))
    plt.axis("off")
    plt.imshow(img)
    plt.show()

def show_img_batch(img_tensor, nrow=8, imgsize=3):
    # [B, C, H ,W]
    img = torchvision.utils.make_grid(img_tensor, nrow=nrow)
    img = img.permute(1, 2, 0)
    ratio = img.shape[1] // img.shape[0]
    plt.figure(figsize=(imgsize * ratio, imgsize))
    plt.axis("off")
    plt.imshow(img)
    plt.show()

def show_idx(model, idx):
    img_t = model.decode_to_img(idx)[0]
    show_img(to_pil(img_t))

def img_loader(img_pth):
    return Image.open(img_pth).convert("RGB")