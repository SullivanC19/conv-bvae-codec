#!/usr/bin/env python3

import sys
import getopt

from tqdm import tqdm
import torch
import torchvision.utils
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split

from model import ConvBetaVae


IMAGE_SIZE = 64
EPOCHS = 100
BATCH_SIZE = 16
DEFAULT_LATENT = 16
DEFAULT_BETA = 1

DIR_DATA = './res/data'
DIR_MODEL = './models'

def load_image_data(img_dir, batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    dataset = datasets.ImageFolder(
        root=img_dir,
        transform=transform
    )

    test_dataset, train_dataset, valid_dataset = random_split(dataset, [2000, 6000, len(dataset) - 8000], generator=torch.Generator().manual_seed(42))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return test_loader, train_loader, valid_loader

def evaluate_model(model, data_loader, optimizer=None):
    tot_loss = 0
    tot_rec = 0
    tot_kl = 0
    for data, _ in tqdm(data_loader):
        if torch.cuda.is_available():
            data = data.cuda()

        loss, rec, kl = model.loss(data)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tot_loss += loss.item()
        tot_rec += rec.item()
        tot_kl += kl.item()

    return tot_loss / len(data_loader), tot_rec / len(data_loader), tot_kl / len(data_loader)

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hl:b:', ['help', 'latent=', 'beta='])
    except getopt.GetoptError as err:
        print(err)
        print('trainer.py -l <latent-variables> -b <beta-value>')
        sys.exit(2)

    latent_variables = DEFAULT_LATENT
    beta_value = DEFAULT_BETA
    for o, a in opts:
        if o in ('-h', '--help'):
            print('trainer.py -l <latent-variables> -b <beta-value>')
            sys.exit()
        elif o in ("-l", "--latent"):
            latent_variables = int(a)
        elif o in ("-b", "--beta"):
            beta_value = int(a)
        else:
            assert False, "unhandled option"

    model = ConvBetaVae(latent_variables, beta_value)
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    test, train, valid = load_image_data(DIR_DATA)

    model_str = f'ConvBetaVae_{latent_variables}_{beta_value}'
    writer = SummaryWriter(f'./runs/{model_str}/')

    for epoch in range(EPOCHS):

        train_loss, train_rec, train_kl = evaluate_model(model, train, optimizer=optimizer)
        valid_loss, valid_rec, valid_kl = evaluate_model(model, valid)

        # Save Loss
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Reconstruction Cost/train', train_rec, epoch)
        writer.add_scalar('KL Divergence/train', train_kl, epoch)
        
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Reconstruction Cost/valid', valid_rec, epoch)
        writer.add_scalar('KL Divergence/valid', valid_kl, epoch)

        # Save Image(s)
        sampled_image = (model.decoder(model.sample_prior(1)) + 1) / 2
        writer.add_image('Sampled Image', torchvision.utils.make_grid(sampled_image, nrow=10), epoch)

        print(f'Epoch {epoch + 1} \t Training Loss: {train_loss}; {train_rec}; {train_kl} \t Validation Loss: {valid_loss}; {valid_rec}; {valid_kl}')
    
        print("Saving Model...")
        torch.save(model, f'{DIR_MODEL}/{model_str}.pkl')
        print("Model Saved")
    
    writer.close()
