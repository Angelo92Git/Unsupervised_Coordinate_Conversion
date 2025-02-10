import torch.nn as nn

cfg_name = 'original_cfg'

class data_cfg:
    data_path = './ee_mumu.csv'

class model_cfg:
    input_dim = 3
    encoding_dim = 2
    encoder_layers = [nn.Linear(input_dim, 128), nn.Sigmoid(), 
                      nn.Linear(128, 64),        nn.Sigmoid(),
                      nn.Linear(64, 32),         nn.Sigmoid(),
                      nn.Linear(32, encoding_dim)]
    decoder_layers = [nn.Linear(encoding_dim, 32), nn.Sigmoid(),
                      nn.Linear(32, 64),           nn.Sigmoid(),
                      nn.Linear(64, 128),          nn.Sigmoid(),
                      nn.Linear(128, input_dim)]
    
class optim_cfg:
    lr = 0.001

class train_cfg:
    batch_size = 32
    num_epochs = 10
    sub_epoch_save_freq = 1
    epoch_save_pace = 1
    best_loss_path = './best_loss/best_loss_128s-64s-32s-2.pkl'
    model_path = './trained_models/model_128s-64s-32s-2.pth'
    latent_path = './latent_data/latent_data_128s-64s-32s-2.pkl'
    latent_viz_path = './latent_viz/latent_viz_128s-64s-32s-2.png'
    log_path = './logs/log_128s-64s-32s-2.log'


