import os
import os.path

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import timm
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

total_epoch = 100  # total epoch
lr_list = [0.0001, 0.0009, 0.008]  # initial learning rate
weight_decay = 0
random_seed_to = 0
angles = [15 * deg for deg in range(24)]


class RotateConcat:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, image):
        rotated_images = []
        for angle in self.angles:
            rotated_image = image.rotate(angle)
            rotated_images.append(rotated_image)

        concatenated_image = np.concatenate([np.expand_dims(img, 0) for img in rotated_images])
        return concatenated_image


def plot_loss_values(tr_loss_arr, val_loss_arr, lr, seed_num, dt_now):
    plt.figure(figsize=(8, 6))
    plt.title(
        f'Loss curve\nTrain loss: {round(tr_loss_arr[0], 3)} --> {round(tr_loss_arr[-1], 3)}\nValidation loss: {round(val_loss_arr[0], 3)} --> {round(val_loss_arr[-1], 3)}')
    plt.plot(range(0, total_epoch), tr_loss_arr, '-b', label="Training loss")
    plt.plot(range(0, total_epoch), val_loss_arr, '-r', label="Validation loss")
    plt.ylim(-0.01, 1.2 * max(max(tr_loss_arr), max(val_loss_arr)))
    plt.legend()

    if not os.path.exists(f'./{dt_now.strftime("%Y%m%d")}/loss_fig'):
        os.makedirs(f'./{dt_now.strftime("%Y%m%d")}/loss_fig')

    plt.savefig(
        f'./{dt_now.strftime("%Y%m%d")}/loss_fig/rndsd_{seed_num}ep{total_epoch}_lr{lr}_l2l{weight_decay}_dt{dt_now.strftime("%Y%m%d%H%M%S")}.png')


def plot_acc_values(tr_acc_arr, val_acc_arr, lr, seed_num, dt_now):
    plt.figure(figsize=(8, 6))
    plt.title(
        f'Accuracy curve\nTrain Accuracy: {round(tr_acc_arr[0], 3)} --> {round(tr_acc_arr[-1], 3)}\nValidation Accucracy: {round(val_acc_arr[0], 3)} --> {round(val_acc_arr[-1], 3)}')
    plt.plot(range(0, total_epoch), tr_acc_arr, '-b', label="Training Accuracy")
    plt.plot(range(0, total_epoch), val_acc_arr, '-r', label="Validation Accuracy")
    plt.ylim(0, 1.2 * max(max(tr_acc_arr), max(val_acc_arr)))
    plt.legend()

    if not os.path.exists(f'./{dt_now.strftime("%Y%m%d")}/acc_fig'):
        os.makedirs(f'./{dt_now.strftime("%Y%m%d")}/acc_fig')

    plt.savefig(
        f'./{dt_now.strftime("%Y%m%d")}/acc_fig/rndsd_{seed_num}_ep{total_epoch}_lr{lr}_l2l{weight_decay}_dt{dt_now.strftime("%Y%m%d%H%M%S")}.png')


for lr_ in lr_list:
    dt_now = datetime.datetime.now()
    lr = lr_  # 0.003

    # fix random seed
    seed_number = 0
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Define the data transforms
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # transforms.Lambda(lambda img: torch.stack([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(transforms.CenterCrop(192)(img.rotate(angle)))) for angle in angles])),
        transforms.RandomHorizontalFlip(p=0.7),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=16)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=16)

    # Define the ResNet-18 model with pre-trained weights
    model = timm.create_model('resnet18', pretrained=True, num_classes=10)
    model = model.to(device)  # Move the model to the GPU

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    # Define the learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch * 0.2, eta_min=1e-6)


    def train():
        model.train()

        for i, data in enumerate(trainloader, 0):
            if i % 100 == 0:
                running_loss = 0.0
                correct = 0.0
                total = 0

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move the input data to the GPU
            # labels = torch.unsqueeze(labels, 0).repeat((24, 1)).T.flatten()
            # bs, rot_img, c, h, w = inputs.size()
            optimizer.zero_grad()
            # outputs = model(inputs.view(-1, c, h, w))
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                acc_calc = 100 * correct / total
                loss_calc = running_loss / total
                print(f'epoch: {epoch + 1}, iteration: {i + 1}, acc: {acc_calc}, loss: {loss_calc}')
        return loss_calc, acc_calc

    def test():
        model.eval()

        # Test the model
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)  # Move the input data to the GPU
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc_calc = 100 * correct / total
        loss_calc = running_loss / total

        print(
            f'Performance of the network on the 10000 test images, lr: {lr}. l2: {weight_decay},acc: {acc_calc}, loss: {loss_calc} %%')
        return loss_calc, acc_calc


    # Train the model
    tr_loss_list = []
    tr_acc_list = []
    val_loss_list = []
    val_acc_list = []
    for epoch in range(total_epoch):
        tr_loss, tr_acc = train()
        tr_loss_list.append(tr_loss)
        tr_acc_list.append(tr_acc)

        val_loss, val_acc = test()
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        scheduler.step()

        if not os.path.exists(f'./{dt_now.strftime("%Y%m%d")}/ckpt'):
            os.makedirs(f'./{dt_now.strftime("%Y%m%d")}/ckpt')

        PATH = f'./{dt_now.strftime("%Y%m%d")}/ckpt/ctcrop_resnet18_cifar10_rndsd_{seed_number}_lr{lr}_epoch{epoch}_l2l{weight_decay}_dt{dt_now.strftime("%Y%m%d%")}.pth'
        torch.save(model.state_dict(), PATH)

    plot_loss_values(tr_loss_list, val_loss_list, lr, seed_number, dt_now)
    plot_acc_values(tr_acc_list, val_acc_list, lr, seed_number, dt_now)
    print('Finished Training')

    # Save the checkpoint of the last model

    if not os.path.exists(f'./{dt_now.strftime("%Y%m%d")}/ckpt'):
        os.makedirs(f'./{dt_now.strftime("%Y%m%d")}/ckpt')

    PATH = f'./{dt_now.strftime("%Y%m%d")}/ckpt/resnet18_cifar10_rndsd_{seed_number}_lr{lr}_l2l{weight_decay}_dt{dt_now.strftime("%Y%m%d%H%M%S")}.pth'
    torch.save(model.state_dict(), PATH)