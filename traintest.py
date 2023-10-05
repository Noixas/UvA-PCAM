import os
import argparse
import torch
from pcam import get_dataloaders, get_model, train, test


# Parameters
parser = argparse.ArgumentParser(description="Train+Test script",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-model",  choices=['AlexNet', 'VGG-16', 'VGG-11', 'GoogleNet', 'Inception-v3',
                                        'ResNet-18', 'DenseNet-161', 'ViT-Base-16'], help="Model name")
parser.add_argument("-augment", action='store_true', default=False, help="Add data augmentations or not")
parser.add_argument("-batch", type=int, default=32, help="Batch size")
parser.add_argument("-epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("-classes", type=int, default=2, help="Number of classes")
parser.add_argument("-lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("-save_model", default=None, help="Save checkpoint path")
parser.add_argument("-load_model", default=None, help="Load checkpoint path")
parser.add_argument("-save_results", default=None, help="Load checkpoint path")
args = parser.parse_args()
config = vars(args)
print(config)


# Check if GPU is used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Data
if 'Inception' in config['model']:
    resize = 299
elif 'ViT' in config['model']:
    resize = 224
else:
    resize = 96
train_loader, val_loader, test_loader = get_dataloaders('data', batch_size=config['batch'], resize=resize,
                                                        augment=config['augment'])

# Model
model, params = get_model(config['model'], device)

# Optimizer
optimizer = torch.optim.Adam(params, lr=config['lr'])

# Scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], epochs=config['epochs'],
                                                steps_per_epoch=int(len(train_loader)/config['batch']))
# Loss Function
loss_fun = torch.nn.CrossEntropyLoss()

# Train
train(model, train_loader, val_loader, loss_fun, optimizer, scheduler, num_epochs=config['epochs'],
      num_classes=config['classes'], device=device, save_ckpt_path=config['save_model'],
      load_ckpt_path=config['load_model'])

# Test
if config['load_model'] is None:
    if config['save_model'] is None:
        load_ckpt_path = os.path.join('models',
                                  f'{model.__class__.__name__}_lr{str(optimizer.defaults["lr"]).split(".")[1]}_epoch{config["epochs"]}.pt')
    else:
        load_ckpt_path = config['save_model']
test(model, test_loader, loss_fun, config['classes'], device, load_ckpt_path=load_ckpt_path,
     save_results_path=config['save_results'])
