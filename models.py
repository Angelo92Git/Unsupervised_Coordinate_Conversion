import torch
import torch.nn as nn
from config import model_cfg

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(*model_cfg.encoder_layers)
        
        self.decoder = nn.Sequential(*model_cfg.decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        