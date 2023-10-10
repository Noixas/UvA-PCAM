import os
import numpy as np
import pandas as pd
from tqdm import tqdm as _tqdm
from ptflops import get_model_complexity_info

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import PCAM
import torchvision.transforms.v2 as transforms
from torcheval.metrics import MulticlassAUROC, MulticlassAccuracy

from torchvision.models import alexnet, vgg11, vgg16, googlenet, inception_v3, resnet18, densenet161
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import swin_v2_b

class APCAM(PCAM):
    """
    Augmented PCAM dataset
    """
    _FILES = {
        "train": {
            "images": (
                "camelyonpatch_level_2_split_train_x.h5",  # Data file name
                "1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2",  # Google Drive ID
                "1571f514728f59376b705fc836ff4b63",  # md5 hash
            ),
            "targets": (
                "camelyonpatch_level_2_split_train_y.h5",
                "1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG",
                "35c2d7259d906cfc8143347bb8e05be7",
            ),
        },
        "test": {
            "images": (
                "camelyonpatch_level_2_split_test_x.h5",
                "1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_",
                "d8c2d60d490dbd479f8199bdfa0cf6ec",
            ),
            "targets": (
                "camelyonpatch_level_2_split_test_y.h5",
                "17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP",
                "60a7035772fbdb7f34eb86d4420cf66a",
            ),
        },
        "val": {
            "images": (
                "camelyonpatch_level_2_split_valid_x.h5",
                "1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3",
                "d5b63470df7cfa627aeec8b9dc0c066e",
            ),
            "targets": (
                "camelyonpatch_level_2_split_valid_y.h5",
                "1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO",
                "2b85f58b927af9964a4c15b8f7e8f179",
            ),
        },
        "train-augment": {
            "images": (
                "camelyonpatch_level_2_split_augment_train_x.h5",
                "",
                "",
            ),
            "targets": (
                "camelyonpatch_level_2_split_augment_train_y.h5",
                "",
                "",
            ),
        },
        "test-augment": {
            "images": (
                "camelyonpatch_level_2_split_augment_test_x.h5",
                "",
                "",
            ),
            "targets": (
                "camelyonpatch_level_2_split_augment_test_y.h5",
                "",
                "",
            ),
        },
        "val-augment": {
            "images": (
                "camelyonpatch_level_2_split_augment_valid_x.h5",
                "",
                "",
            ),
            "targets": (
                "camelyonpatch_level_2_split_augment_valid_y.h5",
                "",
                "",
            ),
        },
    }


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


def get_dataloaders(data_path, batch_size, shuffle=True, download=True, resize=96, augment=False):
    """
    Creates dataloaders from dataset
    """

    # Preprocessing
    preprocess_list = [
        transforms.PILToTensor(),
        transforms.Resize(resize, antialias=True)
    ]

    # Data augmentations
    if augment is True:
        augment_list = [
            transforms.RandomResizedCrop(resize, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter()
        ]
    else:
        augment_list = []

    # Normalization
    normalize_list = [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    train_transform = transforms.Compose(preprocess_list + augment_list + normalize_list)  # Apply data augments only in train
    testval_transform = transforms.Compose(preprocess_list + normalize_list)

    print(f'Train Transforms:')
    print(train_transform)

    train_dataset = PCAM(root=data_path, split='train', download=download, transform=train_transform)
    val_dataset = PCAM(root=data_path, split='val', download=download, transform=testval_transform)
    test_dataset = PCAM(root=data_path, split='test', download=download, transform=testval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader


def get_model(model_name, device):
    model_dir = {'AlexNet': alexnet,
                 'VGG-16': vgg16,
                 'VGG-11': vgg11,
                 'GoogleNet': googlenet,
                 'Inception-v3': inception_v3,
                 'ResNet-18': resnet18,
                 'DenseNet-161': densenet161,
                 'ViT-Base-16': vit_b_16,
                 'Swin-V2-Base': swin_v2_b}

    model = model_dir[model_name](pretrained=True)
    model.to(device)
    print(f'Selected Model: {model.__class__.__name__}')

    # Freeze all layers except last
    for param in model.parameters():
        param.requires_grad = False

    # Create classification layer
    num_classes = 2
    if model.__class__.__name__ in ['AlexNet', 'VGG']:
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model.__class__.__name__ == 'DenseNet':
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    elif model.__class__.__name__ == 'VisionTransformer':
        model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
    elif model.__class__.__name__ == 'SwinTransformer':
        model.head = torch.nn.Linear(model.head.in_features, num_classes)
    else:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    return model


def train(model, train_loader, val_loader, loss_fun, optimizer, scheduler, num_epochs, num_classes, device, save_ckpt_path=None, load_ckpt_path=None, logger=None, run=None):
    """
    Trains model
    """

    model.to(device)

    # Start from checkpoint
    if load_ckpt_path is not None:
        checkpoint = torch.load(load_ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])

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
        
        i = 0
        # Train
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}, Training'):
            # Move the inputs and labels to the device
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            # Zero the optimizer gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fun(outputs, labels)


            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

            # Update the running loss and metrics
            loss_arr.append(loss.item())
            auc.update(outputs, labels)
            accuracy.update(outputs, labels)
            
            # Log metrics
            # Log after every 30 steps
            if i % 100 == 0 and run != None:
                run[logger.base_namespace]["batch/loss"].append(loss.item())
                run[logger.base_namespace]["batch/acc"].append( accuracy.compute())
                run[logger.base_namespace]["batch/auc"].append( auc.compute())
            i+=1
        # Scheduler step
        scheduler.step()
        
        # Calculate the train loss and metrics
        train_loss = np.average(loss_arr)
        train_acc = accuracy.compute()
        train_auc = auc.compute()

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
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = loss_fun(outputs, labels)

                # Update the running loss and metrics
                loss_arr.append(loss.item())
                auc.update(outputs, labels)
                accuracy.update(outputs, labels)

        # Calculate the validation loss, accuracy and AUC
        val_loss = np.average(loss_arr)
        val_acc = accuracy.compute()
        val_auc = auc.compute()

        # Print the epoch results
        print(
            'Train Loss: {:.4f}, Train Acc: {:.4f}, Train AUC: {:.4f}, \n Val Loss: {:.4f}, Val Acc: {:.4f}, Val AUC: {:.4f}'
            .format(train_loss, train_acc, train_auc, val_loss, val_acc, val_auc))

    # Save model
    if save_ckpt_path is None:
        save_ckpt_path = os.path.join('models',
                                      f'{model.__class__.__name__}_lr{str(optimizer.defaults["lr"]).split(".")[1]}_epoch{num_epochs}.pt')
        if not os.path.exists('models'):  # If folder 'models' doesn't exist, create it
            os.makedirs('models')
    save_ckpt_path = uniquify(
        save_ckpt_path)  # Create unique path name by appending number if given path already exists
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


def test(model, test_loader, loss_fun, num_classes, device, load_ckpt_path=None, save_results_path=None):
    """
    Tests model
    """

    if load_ckpt_path is not None:
        checkpoint = torch.load(load_ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.to(device)

    # Create metric monitors
    auc = MulticlassAUROC(num_classes=num_classes)
    accuracy = MulticlassAccuracy()

    model.eval()  # Set the model to evaluation mode

    # Initialize the running loss and metrics
    loss_arr = []
    auc.reset()
    accuracy.reset()

    ## Test
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            # Move the inputs and labels to the device
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fun(outputs, labels)

            # Update the running loss and metrics
            loss_arr.append(loss.item())
            auc.update(outputs, labels)
            accuracy.update(outputs, labels)

    # Calculate the test loss, accuracy and AUC
    test_loss = np.average(loss_arr)
    test_acc = accuracy.compute().detach().numpy()
    test_auc = auc.compute().detach().numpy()

    # Calculate GFLOPS
    image_size = tuple(next(iter(test_loader))[0].shape[1:])
    macs, _ = get_model_complexity_info(model, image_size,
                                        as_strings=False, print_per_layer_stat=False, verbose=False)
    gflops = 2 * macs / 1000000000

    # Print the test results
    print('GFLOPS: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}, Test AUC: {:.4f}'.format(gflops, test_loss, test_acc, test_auc))

    ## Save results
    results = pd.DataFrame({'model': model.__class__.__name__, 'gflops': [gflops], 'test_loss': [test_loss], 'test_acc': [test_acc], 'test_auc': [test_auc]})
    if save_results_path is None:
        if load_ckpt_path is not None:
            save_results_path = load_ckpt_path.split('.')[0] + '.csv'
        else:
            save_results_path = os.path.join('models', f'{model.__class__.__name__}.csv')
            if not os.path.exists('models'):  # If folder 'models' doesn't exist, create it
                os.makedirs('models')
    save_results_path = uniquify(
        save_results_path)  # Create unique path name by appending number if given path already exists
    results.to_csv(save_results_path)
    print(f'Saved results at: {save_results_path}\n')
