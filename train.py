import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle as pkl
import os
import logging

from config import data_cfg, model_cfg, optim_cfg, train_cfg
from models import Autoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.basicConfig(level=logging.DEBUG, filename=train_cfg.log_path, filemode='a+', format='%(asctime)-15s %(levelname)-8s %(message)s')

df = pd.read_csv(data_cfg.data_path)
data = df[['# px_minus', 'py_minus', 'pz_minus']].values.astype(np.float32)
data_tensor = torch.tensor(data, device=device)

best_loss = 999999

if os.path.isfile(train_cfg.best_loss_path):
    best_loss = pkl.load(open(train_cfg.best_loss_path, 'rb'))

model = Autoencoder(model_cfg.input_dim, model_cfg.encoding_dim)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=optim_cfg.lr)
criterion = nn.MSELoss()

if os.path.isfile(train_cfg.model_path):
    model.load_state_dict(torch.load(train_cfg.model_path))

dataset = TensorDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True)


for epoch in range(train_cfg.num_epochs):
    for batch_id, batch in enumerate(dataloader):
        input = batch[0]
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, input)
        loss.backward()
        optimizer.step()
        if ((batch_id+1) % (len(dataloader) // (train_cfg.sub_epoch_save_freq)) == 0) and (epoch % train_cfg.epoch_save_pace == 0):
            with torch.no_grad():
                total_loss = criterion(model(data_tensor), data_tensor)
                if total_loss < best_loss:
                    best_loss = total_loss
                    torch.save(model.state_dict(), train_cfg.model_path)
                    logging.info(f'Saving model at epoch {epoch} and batch {batch_id}')
                    print(f'Saving model at epoch {epoch} and batch {batch_id}')
                    pkl.dump(best_loss, open(train_cfg.best_loss_path, 'wb'))
                    encoded_data = model.encoder(data_tensor).numpy()
                    pkl.dump(encoded_data, open(train_cfg.latent_path, 'wb'))
                    random_indices = np.random.choice(range(len(encoded_data)), 1000)
                    plt.figure()
                    plt.scatter(encoded_data[:, 0], encoded_data[:, 1])
                    plt.savefig(train_cfg.latent_viz_path)
    
    logging.info(f'Epoch [{epoch+1}/{train_cfg.num_epochs}] loss: {loss.item():.2f}')
    print(f'Epoch [{epoch+1}/{train_cfg.num_epochs}] loss: {loss.item():.2f}')
