import argparse
import torch
from pcam import get_dataloaders, get_model, train, test

# Optimization
torch.backends.cudnn.benchmark = True

# Parameters
parser = argparse.ArgumentParser(description="Test script",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-model",  choices=['AlexNet', 'VGG-16', 'VGG-11', 'GoogleNet', 'Inception-v3',
                                        'ResNet-18', 'DenseNet-161', 'Swin-v2-Base','Swin-v2-Base-micro'], help="Model name")
parser.add_argument("-test_runs", type=int, default=1, help="Number of testing repetitions (to quantify uncertainty)")
parser.add_argument("-batch", type=int, default=256, help="Batch size")
parser.add_argument("-epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("-classes", type=int, default=2, help="Number of classes")
parser.add_argument("-load_model", default=None, help="Path to load checkpoint")
parser.add_argument("-data_path", default='data', help="Path to load data from")

args = parser.parse_args()
config = vars(args)
print(f'Arguments: {config}')


# Check if GPU is used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# Data
if 'Inception' in config['model']:
    resize = 299
elif 'Swin-v2-Base-micro' in config['model']:
    resize = 256
else:
    resize = 96
_, _, test_loader = get_dataloaders(config['data_path'], resize=resize,batch_size=config['batch'], train=False)

# Model
model = get_model(config['model'], device)

# Loss Function
loss_fun = torch.nn.CrossEntropyLoss()

# If we run the model for multiple runs, then quantify uncertainty using dropout (set dropout to True)
dropout = config['test_runs'] > 1

# Test
for i in range(config['test_runs']):
    print(f'Testing Run {i+1}/{config["test_runs"]}')
    test(model, test_loader, loss_fun, config['classes'], device, load_ckpt_path=config['load_model'], dropout=dropout)
