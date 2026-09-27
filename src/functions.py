#!/usr/bin/env python
# coding: utf-8

# Importing modules
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# DL modules
import pandas as pd
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, asdict


def rbf_1d(x_0, centroids, sigma):
    """
    Compute the (unnormalized) RBF weights:
      w_k = exp( -||x_0 - c_k||^2 / (2 sigma^2) )
    """
    # squared distances to each centroid
    d2 = np.sum((x_0 - centroids)**2, axis=1)
    # Gaussian kernel
    return np.exp(-d2 / (2*sigma**2))


def forcing_from_rbf(z_t, centroids, actions, sigma):
    """
    Normalized RBF interpolation:
      b(z_t) = sum_k w_k * b_k   with  w_k  :=  exp(-||z-c_k||^2/(2σ^2)) / sum_j exp(-||z-c_j||^2/(2σ^2))
    returns a scalar in the convex hull of the actions[], so if actions in [0,0.1] -> output in [0,0.1].
    """
    # pull out numpy array from tensor if needed
    if hasattr(z_t, "cpu"):
        x = z_t.cpu().detach().squeeze().numpy()
    else:
        x = np.asarray(z_t)

    # compute unnormalized weights
    w = rbf_1d(x, centroids, sigma)
    W = np.sum(w)
    if W > 0:
        w /= W
    # dot with your actions
    return float(np.dot(w, actions.flatten()))


# Function to load and process a single file
H1, H2 = 87, 215
W1, W2 = 46, 366

def load_file(file_path):
    img = Image.open(file_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    cropped = arr[H1:H2, W1:W2, :]   # (H, W, 3)
    return cropped


# Custom Dataset Class
class MyDataset(Dataset):
    def __init__(self, data_files):
        self.data_files = data_files

    def __getitem__(self, index): # dunder method 
        file_path = self.data_files[index]
        image = load_file(file_path)
        return image

    def __len__(self):
        return len(self.data_files)
