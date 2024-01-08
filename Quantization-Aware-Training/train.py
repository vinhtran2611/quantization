import os
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import time
import copy
import numpy as np

import wandb

from resnet import resnet18


def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def prepare_dataloader(num_workers=8,
                       train_batch_size=128,
                       eval_batch_size=256):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    train_set = torchvision.datasets.CIFAR10(root="data",
                                             train=True,
                                             download=True,
                                             transform=train_transform)
    # We will use test set for validation and test in this project.
    # Do not use test set for validation in practice!
    test_set = torchvision.datasets.CIFAR10(root="data",
                                            train=False,
                                            download=True,
                                            transform=test_transform)

    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(test_set)

    train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                               batch_size=train_batch_size,
                                               sampler=train_sampler,
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=eval_batch_size,
                                              sampler=test_sampler,
                                              num_workers=num_workers)

    return train_loader, test_loader


def evaluate_model(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def train_model(model,
                train_loader,
                test_loader,
                device,
                learning_rate=1e-1,
                num_epochs=200):

    # The training configurations were not carefully selected.

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=0.9,
                          weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[100, 150],
                                                     gamma=0.1,
                                                     last_epoch=-1)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    best_accuracy = 0.0  # Track the best validation accuracy

    # Evaluation
    model.eval()
    eval_loss, eval_accuracy = evaluate_model(model=model,
                                              test_loader=test_loader,
                                              device=device,
                                              criterion=criterion)
    print("Epoch: {:03d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
        0, eval_loss, eval_accuracy))


    for epoch in range(num_epochs):

        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Log training metrics
        wandb.log({'Train Loss': train_loss, 'Train Accuracy': train_accuracy}, step=epoch)


        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(model=model,
                                                  test_loader=test_loader,
                                                  device=device,
                                                  criterion=criterion)

        # Set learning rate scheduler
        scheduler.step()

        # Log validation metrics
        wandb.log({'Validation Loss': eval_loss, 'Validation Accuracy': eval_accuracy}, step=epoch)

        print(
            "Epoch: {:03d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}"
            .format(epoch + 1, train_loss, train_accuracy, eval_loss,
                    eval_accuracy))

        # Save the best model checkpoint in WandB
        if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy

            ckpt_file = f"best_acc.pth"
            torch.save(model.state_dict(), ckpt_file)

            artifact_name = f"{wandb.run.id}_context_model"
            at = wandb.Artifact(artifact_name, type="model")
            at.add_file(ckpt_file)
            wandb.log_artifact(at, aliases=[f"epoch_{epoch}"])

    return model

import tempfile
def save_model_to_temp(model):
    temp_dir = tempfile.mkdtemp()
    temp_model_path = os.path.join(temp_dir, 'best_checkpoint.pt')
    save_model(model, temp_dir, 'best_checkpoint.pt')
    return temp_model_path


def measure_inference_latency(model,
                              device,
                              input_size=(1, 3, 32, 32),
                              num_samples=100,
                              num_warmups=10):

    model.to(device)
    model.eval()

    x = torch.rand(size=input_size).to(device)

    with torch.no_grad():
        for _ in range(num_warmups):
            _ = model(x)
    torch.cuda.synchronize()

    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_samples):
            _ = model(x)
            torch.cuda.synchronize()
        end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave


def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)


def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model


def save_torchscript_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)


def load_torchscript_model(model_filepath, device):

    model = torch.jit.load(model_filepath, map_location=device)

    return model


def create_model(num_classes=10, weights='DEFAULT'):
    
    # The number of channels in ResNet18 is divisible by 8.
    # This is required for fast GEMM integer matrix multiplication.
    # model = resnet18(num_classes=num_classes, weights=weights)

    model = torchvision.models.resnet18(weights=weights)
    # # We would use the pretrained ResNet18 as a feature extractor.
    # for param in model.parameters():
    #     param.requires_grad = False

    # Modify the last FC layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model



def main():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f'training_run_{current_time}'

    wandb.init(project='pretrain_resnet18_cifar10', name=run_name)

    # Define training configurations
    training_config = {
        'random_seed': 0,
        'num_classes': 10,
        'train_batch_size': 128,
        'eval_batch_size': 128,
        'learning_rate': 1e-1,
        'num_epochs': 200,
        'model_dir': "saved_models",
        'model_filename': "pretrain_resnet18_cifar10.pt",
        "weights": 'DEFAULT', # or None
    }

    # Log training configurations
    wandb.config.update(training_config)
    
    random_seed = training_config['random_seed']
    num_classes = training_config['num_classes']
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")

    model_dir = training_config['model_dir']
    model_filename = training_config['model_filename']
    model_filepath = os.path.join(model_dir, model_filename)

    set_random_seeds(random_seed=random_seed)

    # Create an untrained model.
    model = create_model(num_classes=num_classes, weights = training_config['weights'])
    print_trainable_parameters(model)

    train_loader, test_loader = prepare_dataloader(num_workers=8,
                                                   train_batch_size=training_config['train_batch_size'],
                                                   eval_batch_size=training_config['eval_batch_size'])

    # Train model.
    print("Training Model...")
    model = train_model(model=model,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        device=cuda_device,
                        learning_rate=training_config['learning_rate'],
                        num_epochs=training_config['num_epochs'])
    # Save model.
    save_model(model=model, model_dir=model_dir, model_filename=model_filename)
    # Load a pretrained model.
    # model = load_model(model=model,
    #                    model_filepath=model_filepath,
    #                    device=cuda_device)
    # # Move the model to CPU since static quantization does not support CUDA currently.
    # model.to(cpu_device)

if __name__ == "__main__":

    main()
