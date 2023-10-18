import argparse
import torch
import neptune
from neptune_pytorch import NeptuneLogger
from pcam import get_dataloaders, get_model, train

# Optimization
torch.backends.cudnn.benchmark = True

# Parameters
parser = argparse.ArgumentParser(description="Train script",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-model",  choices=['AlexNet', 'VGG-16', 'VGG-11', 'GoogleNet', 'Inception-v3',
                                        'ResNet-18', 'DenseNet-161', 'Swin-v2-Base'], help="Model name")
parser.add_argument("-augment", action='store_true', default=False, help="To add data augmentations or not")
parser.add_argument("-batch", type=int, default=256, help="Batch size")
parser.add_argument("-epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("-classes", type=int, default=2, help="Number of classes")
parser.add_argument("-lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("-save_model", default=None, help="Path to save checkpoint")
parser.add_argument("-data_path", default='data', help="Path to load data from")
parser.add_argument("-token", default=None, help="File path containing Neptune API Token")
args = parser.parse_args()
config = vars(args)
print(f'Arguments: {config}')


# Check if GPU is used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Data
if 'Inception' in config['model']:
    resize = 299
if 'Swin' in config['model']:
    resize = 256
else:
    resize = 96
train_loader, val_loader, test_loader = get_dataloaders(config['data_path'], batch_size=config['batch'], resize=resize,
                                                        augment=config['augment'])

# Model
model = get_model(config['model'], device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

# Scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], epochs=config['epochs'],
                                                steps_per_epoch=int(len(train_loader) / config['batch']))
# Loss Function
loss_fun = torch.nn.CrossEntropyLoss()

# Initialize Neptune logger
if config['token'] is not None:
    with open(config['token'], 'r') as file:
        token = file.read()
    run = neptune.init_run(
        custom_run_id=model.__class__.__name__,
        project='UvA-2023/pcam2023',
        api_token=token,
        mode='debug'
    )
    logger = NeptuneLogger(
        run=run,
        model=model,
        log_model_diagram=False,
        log_gradients=False,
        log_parameters=True,
        log_freq=30,
    )
else:
    run = None
    logger = None

# Train
train(model, train_loader, val_loader, loss_fun, optimizer, scheduler, num_epochs=config['epochs'],
      num_classes=config['classes'], augment=config['augment'], device=device, save_ckpt_path=config['save_model'], logger=logger, run=run )

# Stop Neptune logger
if run is not None:
    run.stop()

