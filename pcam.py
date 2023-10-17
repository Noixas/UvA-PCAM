import os
import numpy as np
import pandas as pd
from tqdm import tqdm as _tqdm
from ptflops import get_model_complexity_info

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision.datasets import PCAM
import torchvision.transforms as transforms
from torcheval.metrics import MulticlassAUROC, MulticlassAccuracy

from torchvision.models import (alexnet, AlexNet_Weights,
                                vgg11, VGG11_Weights,
                                vgg16, VGG16_Weights,
                                googlenet, GoogLeNet_Weights,
                                inception_v3, Inception_V3_Weights,
                                resnet18, ResNet18_Weights,
                                densenet161, DenseNet161_Weights,
                                swin_v2_b, Swin_V2_B_Weights)


def uniquify(path):
    """
    Creates unique path name by appending number if given path already exists
    """

    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path


def tqdm(*args, **kwargs):
    """
    Wrapper for loop progress bar
    """

    return _tqdm(*args, **kwargs, mininterval=1)  # Safety, do not overflow buffer


def get_dataloaders(data_path, batch_size, train=True, shuffle=True, download=True, resize=96, augment=False):
    """
    Creates dataloaders from dataset
    """

    # Preprocessing
    preprocess_list = [
        transforms.ToTensor(),
        transforms.Resize(resize, antialias=True)
    ]

    # Data augmentations
    if augment is True:
        augment_list = [
            # transforms.RandomResizedCrop(resize, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ColorJitter()
        ]
    else:
        augment_list = []

    # Normalization
    normalize_list = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    if train:
        train_transform = transforms.Compose(
            preprocess_list + augment_list + normalize_list)  # Apply data augments only in train
        print(f'Train Transforms:')
        print(train_transform)

    testval_transform = transforms.Compose(preprocess_list + normalize_list)

    if train:
        train_dataset = PCAM(root=data_path, split='train', download=download, transform=train_transform)
        val_dataset = PCAM(root=data_path, split='val', download=download, transform=testval_transform)
    test_dataset = PCAM(root=data_path, split='test', download=download, transform=testval_transform)

    if train:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        train_loader = None
        val_loader = None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # Do not shuffle so that uncertainty quantification can work (consistent order across runs)

    return train_loader, val_loader, test_loader


def get_model(model_name, device, all_linears=False):
    model_dir = {'AlexNet': (alexnet, AlexNet_Weights.IMAGENET1K_V1),
                 'VGG-11': (vgg11, VGG11_Weights.IMAGENET1K_V1),
                 'VGG-16': (vgg16, VGG16_Weights.IMAGENET1K_V1),
                 'GoogleNet': (googlenet, GoogLeNet_Weights.IMAGENET1K_V1),
                 'Inception-v3': (inception_v3, Inception_V3_Weights.IMAGENET1K_V1),
                 'ResNet-18': (resnet18, ResNet18_Weights.IMAGENET1K_V1),
                 'DenseNet-161': (densenet161, DenseNet161_Weights.IMAGENET1K_V1),
                 'Swin-v2-Base': (swin_v2_b,Swin_V2_B_Weights.IMAGENET1K_V1)}

    model = model_dir[model_name][0](weights=model_dir[model_name][1])
    model.to(device)
    print(f'Selected Model: {model.__class__.__name__}\n')

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classification layers
    num_classes = 2
    if all_linears:
        if model.__class__.__name__ == 'AlexNet':
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=9216, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=4096, out_features=num_classes, bias=True)
            )
        elif model.__class__.__name__ == 'VGG':
            model.classifier = nn.Sequential(
                nn.Linear(in_features=25088, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(in_features=4096, out_features=num_classes, bias=True),
            )
        elif model.__class__.__name__ == 'DenseNet':
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
        elif model.__class__.__name__ == 'VisionTransformer':
            model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
        elif model.__class__.__name__ == 'SwinTransformer':
            model.head = torch.nn.Linear(model.head.in_features, num_classes)
        else:
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        if model.__class__.__name__ == 'AlexNet':
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        elif model.__class__.__name__ == 'VGG':
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        elif model.__class__.__name__ == 'DenseNet':
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif model.__class__.__name__ == 'VisionTransformer':
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        elif model.__class__.__name__ == 'SwinTransformer':
            model.head = nn.Linear(model.head.in_features, num_classes)
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def train(model, train_loader, val_loader, loss_fun, optimizer, scheduler, num_epochs, num_classes, device,
          augment=False, save_ckpt_path=None, load_ckpt_path=None, logger=None, run=None):
    """
    Trains model
    """

    model.to(device)

    if 'Inception' in model.__class__.__name__:
        model.aux_logits = False

    # Start from checkpoint
    if load_ckpt_path is not None:
        checkpoint = torch.load(load_ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Create loss and metric lists
    train_loss_arr = []
    train_auc_arr = []
    train_acc_arr = []
    val_loss_arr = []
    val_auc_arr = []
    val_acc_arr = []

    # Create metric monitors
    auc = MulticlassAUROC(num_classes=num_classes)
    accuracy = MulticlassAccuracy()

    for epoch in range(num_epochs):

        # Set the model to train mode
        model.train()

        # Initialize the running loss and metrics
        loss_arr = []
        auc.reset()
        accuracy.reset()

        # Train
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}, Training'):
            # Move the inputs and labels to the device
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            # Zero the optimizer gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model(inputs)
            _, preds = torch.max(logits, 1)
            loss = loss_fun(logits, labels)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Update the running loss and metrics
            loss_arr.append(loss.item())
            auc.update(logits, labels)  # AUC handles logits accordingly
            accuracy.update(logits, labels)  # Accuracy too

        # Scheduler step
        scheduler.step()

        # Calculate the loss and metrics
        train_loss = np.average(loss_arr)
        train_acc = accuracy.compute().item()
        train_auc = auc.compute().item()

        # Log the loss and metrics
        train_loss_arr.append(train_loss)
        train_acc_arr.append(train_acc)
        train_auc_arr.append(train_auc)
        if logger is not None and run is not None:
            run[logger.base_namespace]["batch/train_loss"].append(train_loss)
            run[logger.base_namespace]["batch/train_acc"].append(train_acc)
            run[logger.base_namespace]["batch/train_auc"].append(train_auc)

        # Set the model to evaluation mode
        model.eval()

        # Initialize the running loss and metrics
        loss_arr = []
        auc.reset()
        accuracy.reset()

        # Validate
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs}, Validation'):
                # Move the inputs and labels to the device
                inputs = inputs.float().to(device)
                labels = labels.to(device)

                # Forward pass
                logits = model(inputs)
                _, preds = torch.max(logits, 1)
                loss = loss_fun(logits, labels)

                # Update the running loss and metrics
                loss_arr.append(loss.item())
                auc.update(logits, labels)
                accuracy.update(logits, labels)

        # Calculate the validation loss, accuracy and AUC
        val_loss = np.average(loss_arr)
        val_acc = accuracy.compute().item()
        val_auc = auc.compute().item()

        # Log the loss and metrics
        val_loss_arr.append(val_loss)
        val_acc_arr.append(val_acc)
        val_auc_arr.append(val_auc)
        if logger is not None and run is not None:
            run[logger.base_namespace]["batch/val_loss"].append(val_loss)
            run[logger.base_namespace]["batch/val_acc"].append(val_acc)
            run[logger.base_namespace]["batch/val_auc"].append(val_auc)

        # Print the epoch results
        print(
            'Train Loss: {:.4f}, Train Acc: {:.4f}, Train AUC: {:.4f}, \n Val Loss: {:.4f}, Val Acc: {:.4f}, Val AUC: {:.4f}'
            .format(train_loss, train_acc, train_auc, val_loss, val_acc, val_auc))

    # Save model
    if save_ckpt_path is None:
        save_ckpt_folder = os.path.join('models',
                                        f'{model.__class__.__name__}_lr{str(optimizer.defaults["lr"]).split(".")[1]}_epoch{num_epochs}' + ('_augment' if augment else ''))
        save_ckpt_folder = uniquify(
            save_ckpt_folder)  # Create unique folder name by appending number if given path already exists

        save_ckpt_path = os.path.join(save_ckpt_folder, f'{model.__class__.__name__}.pt')

        if not os.path.exists('models'):  # If folder 'models' doesn't exist, create it
            os.makedirs('models')
        if not os.path.exists(save_ckpt_folder):  # If model folder doesn't exist, create it
            os.makedirs(save_ckpt_folder)

    torch.save({
        'epochs': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_auc': train_auc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_auc': val_auc,
    }, save_ckpt_path)
    print(f'Saved checkpoint at: {save_ckpt_path}\n')

    # Save learning curve
    curve = pd.DataFrame(
        {'model': model.__class__.__name__,
         'train_loss': train_loss_arr, 'train_acc': train_acc_arr, 'train_auc': train_auc_arr,
         'val_loss': val_loss_arr, 'val_acc': val_acc_arr, 'val_auc': val_auc_arr}
    )
    save_curve_path = save_ckpt_path.split('.')[0] + '_curve.csv'
    curve.to_csv(save_curve_path)
    print(f'Saved curve at: {save_curve_path}\n')

    return save_ckpt_path


def test(model, test_loader, loss_fun, num_classes, device, dropout=False, load_ckpt_path=None):
    """
    Tests model
    """

    # Load model checkpoint
    if load_ckpt_path is not None:
        checkpoint = torch.load(load_ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)

    # Create metric monitors
    auc = MulticlassAUROC(num_classes=num_classes)
    accuracy = MulticlassAccuracy()

    # Set the model to evaluation mode
    model.eval()

    # If dropout is enabled then make the dropout layers trainable
    if dropout:
        for module in model.modules():
            if 'Dropout' in module.__class__.__name__:
                module.train()

    # Initialize the running loss and metrics
    loss_arr = []
    auc.reset()
    accuracy.reset()

    # Initialize prediction and label list to save as file later
    pos_probs_list = []  # Probabilities for positive class
    neg_probs_list = []  # Probabilities for negative class
    labels_list = []

    # Test
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            # Move the inputs and labels to the device
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            # Forward pass
            logits = model(inputs)
            _, preds = torch.max(logits, 1)
            loss = loss_fun(logits, labels)

            # Update the running loss and metrics
            loss_arr.append(loss.item())
            auc.update(logits, labels)
            accuracy.update(logits, labels)

            # Update the list of all predictions and labels
            probs = f.softmax(logits, dim=1)
            pos_probs_list += probs[:, 1].detach().tolist()
            neg_probs_list += probs[:, 0].detach().tolist()
            labels_list += labels.detach().tolist()

    # Calculate the test loss, accuracy and AUC
    test_loss = np.average(loss_arr)
    test_acc = accuracy.compute().detach().numpy()
    test_auc = auc.compute().detach().numpy()

    # Calculate GFLOPS
    image_size = tuple(next(iter(test_loader))[0].shape[1:])
    macs, _ = get_model_complexity_info(model, image_size,
                                        as_strings=False, print_per_layer_stat=False, verbose=False)
    gflops = 2 * macs / 1000000000

    # Print the test metrics
    print('GFLOPS: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, Test AUC: {:.4f}'.format(gflops, test_loss, test_acc,
                                                                                         test_auc))

    # Save outputs and metrics
    outputs = pd.DataFrame({'pos_probs': pos_probs_list, 'neg_probs': neg_probs_list, 'labels': labels_list})
    metrics = pd.DataFrame(
        {'model': model.__class__.__name__, 'gflops': [gflops], 'test_loss': [test_loss], 'test_acc': [test_acc],
         'test_auc': [test_auc]})
    if load_ckpt_path is not None:
        save_outputs_path = uniquify(load_ckpt_path.split('.')[0] + '_outputs.csv')  # Create unique file
        save_metrics_path = uniquify(load_ckpt_path.split('.')[0] + '_metrics.csv')
    else:
        save_folder = os.path.join('models', f'{model.__class__.__name__}')

        save_outputs_path = uniquify(os.path.join(save_folder, f'{model.__class__.__name__}_outputs.csv'))
        save_metrics_path = uniquify(os.path.join(save_folder, f'{model.__class__.__name__}_metrics.csv'))

        if not os.path.exists('models'):  # If folder 'models' doesn't exist, create it
            os.makedirs('models')
        if not os.path.exists(save_folder):  # If model folder doesn't exist, create it
            os.makedirs(save_folder)

    outputs.to_csv(save_outputs_path)
    metrics.to_csv(save_metrics_path)
    print(f'Saved outputs at: {save_metrics_path}\n')
    print(f'Saved metrics at: {save_metrics_path}\n')
