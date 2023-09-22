import os
import numpy as np
import pandas as pd
from tqdm import tqdm as _tqdm
from ptflops import get_model_complexity_info

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import PCAM
import torchvision.transforms as transforms
from torcheval.metrics import MulticlassAUROC, MulticlassAccuracy


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


def get_dataloaders(data_path, batch_size, shuffle=True, download=True):
    """
    Creates dataloaders from dataset
    """

    # Create a transform to convert all images to tensors
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    train_dataset = PCAM(root='data', split='train', download=True, transform=transform)
    val_dataset = PCAM(root='data', split='val', download=True, transform=transform)
    test_dataset = PCAM(root='data', split='test', download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def train(model, train_loader, val_loader, loss_fun, optimizer, num_epochs, num_classes, device, save_ckpt_path=None):
    """
    Trains model
    """

    model.to(device)

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

        ## Train
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

        ## Validate
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
            'Train Loss: {:.4f}, Train Acc: {:.4f}, Train AUC: {:.4f}, \n Val Loss: {:.4f}, Val Acc: {:.4f}, Val AUC: {:.4f}\n'
            .format(train_loss, train_acc, train_auc, val_loss, val_acc, val_auc))

        ## Save model
        if save_ckpt_path is None:
            save_ckpt_path = os.path.join('models',
                                          f'{model.__class__.__name__}_lr{str(optimizer.defaults["lr"]).split(".")[1]}_epoch{epoch}.pt')
            if not os.path.exists('models'):  # If folder 'models' doesn't exist, create it
                os.makedirs('models')
        save_ckpt_path = uniquify(
            save_ckpt_path)  # Create unique path name by appending number if given path already exists
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_auc': train_auc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc,
        }, save_ckpt_path)
        print(f'Saved checkpoint at: {save_ckpt_path}')


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
    print(f'Saved results at: {save_results_path}')