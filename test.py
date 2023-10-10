import argparse
import torch
from pcam import get_dataloaders, get_model, train, test

#Optimization
torch.backends.cudnn.benchmark = True

# Parameters
parser = argparse.ArgumentParser(description="Test script",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-model",  choices=['AlexNet', 'VGG-16', 'VGG-11', 'GoogleNet', 'Inception-v3',
                                        'ResNet-18', 'DenseNet-161', 'ViT-Base-16'], help="Model name")
parser.add_argument("-batch", type=int, default=32, help="Batch size")
parser.add_argument("-classes", type=int, default=2, help="Number of classes")
parser.add_argument("-load_model", help="Load checkpoint path")
parser.add_argument("-save_results", default=None, help="Save results path")
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
    resize = None
train_loader, val_loader, test_loader = get_dataloaders('data', batch_size=config['batch'], resize=resize)

# Model
model, params = get_model(config['model'], device)

# Loss Function
loss_fun = torch.nn.CrossEntropyLoss()

# Test
test(model, test_loader, loss_fun, config['classes'], device, load_ckpt_path=config['load_model'],
     save_results_path=config['save_results'])
