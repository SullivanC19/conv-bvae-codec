#!/usr/bin/env python3

import sys
import getopt
from numpy.lib.function_base import interp

from sklearn import svm
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, utils

from trainer import load_image_data, evaluate_model, DIR_DATA, DIR_MODEL

def load_model(latent_variables=32, beta_value=1):
    model_str = f'ConvBetaVae_{latent_variables}_{beta_value}'
    return torch.load(f'{DIR_MODEL}/{model_str}.pkl', map_location=torch.device('cpu'))

def load_all_models():
    all_models = []
    for i in [10, 20, 30, 40, 50]:
        models = []
        for j in [1, 4, 8, 12, 16, 20]:
            models.append(load_model(i, j))
        all_models.append(models)
    return all_models

# used to check performance on test set
def compute_test_loss(model, img_dir):
    test, _, _ = load_image_data(img_dir)
    return evaluate_model(model, test)

# used to check sample quality
def compute_psnr(model, img_dir):
    test, _, _ = load_image_data(img_dir)
    mse = torch.nn.MSELoss()
    tot_psnr = 0
    for data, _ in tqdm(test):
        if torch.cuda.is_available():
            data = data.cuda()
        
        prediction = model.forward(data)
        err = mse(prediction, data)
        tot_psnr += 10 * torch.log10(4 / err)

    return tot_psnr / len(test)

# used to check how normal the compressed 2-dimensional distribution looks
def plot_pca(model, img_dir):
    test, _, _ = load_image_data(img_dir)
    A = torch.zeros((len(test), model.latent_variables))

    
    for i, (data, _) in enumerate(tqdm(test)):
        if torch.cuda.is_available():
            data = data.cuda()
        A[i] = model.encoder(data)[0]

    U, _, _ = torch.pca_lowrank(A)
    U = U.detach().numpy()
    plt.scatter(U[:, 0], U[:, 1], s=6)
    plt.show()
    
# used to check how normal the compressed 2-dimensional distribution looks
def plot_all_pca(models, img_dir):
    test, _, _ = load_image_data(img_dir)
    A = []
    for i in range(len(models)):
        _A = []
        for j in range(len(models[0])):
            _A.append(torch.zeros((len(test), models[i][j].latent_variables)))
        A.append(_A)

    figure, axis = plt.subplots(len(models), len(models[0]))
    for k, (data, _) in enumerate(tqdm(test)):
        for i in range(len(models)):
            for j in range(len(models[0])):
                A[i][j][k] = models[i][j].encoder(data)[0]
    
    for i in range(len(models)):
        for j in range(len(models[0])):   
            U, _, _ = torch.pca_lowrank(A[i][j])
            U = U.detach().numpy()
            axis[i][j].scatter(U[:, 0], U[:, 1], s=.5)
            axis[i][j].set_xticks([])
            axis[i][j].set_yticks([])
            axis[i][j].set_xlim([-.07, .07])
            axis[i][j].set_ylim([-.07, .07])

    plt.show()

# used to show reconstruction quality
def show_side_to_side_comparison(model, img_dir, num_samples=10):
    imgs = torch.zeros((num_samples * 2, 3, 64, 64))
    test, _, _ = load_image_data(img_dir)
    for i, (data, smile_label) in enumerate(test):
        if i == num_samples:
            break
        imgs[i * 2] = (data[0] + 1) / 2
        imgs[i * 2 + 1] = (model.forward(data)[0] + 1) / 2
    
    plt.imshow(utils.make_grid(imgs, nrow=2).permute(1, 2, 0))
    plt.show()

# used to show flexibility of latent space encoding
def show_smile_translation(model, img_dir, num_samples=5, num_across=15):
    test, _, _ = load_image_data(img_dir)
    A = torch.zeros((len(test), model.latent_variables))
    labels = np.zeros(len(test))

    imgs = torch.zeros(num_samples, 3, 64, 64)
    for i, (data, smile_label) in enumerate(tqdm(test)):
        if i < num_samples: # TODO choose a good image for demonstration
            imgs[i] = data[0]
        A[i] = model.encoder(data)[0]
        labels[i] = smile_label

    # calculate smile vector
    smiles = np.where(labels == 1)
    no_smiles = np.where(labels == 0)
    A = A
    v = A[smiles].sum(0) - A[no_smiles].sum(0)
    v = (v / torch.norm(v)).expand(num_samples, v.size(0))

    z, _ = model.encoder(imgs)
    interpolated = []
    for i in range(num_across):
        if i == num_across // 2:
            interpolated.append((imgs + 1) / 2)
            continue
        z_hat = z + v * (i - num_across // 2) / 3
        interpolated.append((model.decoder(z_hat) + 1) / 2)

    interpolated = torch.cat(interpolated)
    
    plt.imshow(utils.make_grid(interpolated, nrow=num_samples, ).permute(1, 2, 0))
    plt.show()


