"""
Feel free to replace this code with your own model training code. 
This is just a simple example to get you started.

This training script uses imports relative to the base directory (assignment/).
To run this training script with uv, ensure you're in the root directory (assignment/)
and execute: uv run -m submission.fashion_training
"""
import os
import numpy as np
import torch, torchvision

from submission import engine
from submission.fashion_model import Net

# For plotting the graphs
train_loss_history = []
val_loss_history = []
val_acc_history = []

def get_data_loaders(batch_size=64, val_fraction=0.1):
    """
    Create train, validation and test DataLoaders for Fashion-MNIST.
    Uses a deterministic train/val split of the official training set
    and a separate test set. Transforms are shared with get_transforms()
    so that training and marking use consistent preprocessing.
    """

   # Loading transforms for training and evaluation
    train_transform = get_transforms(mode='train')
    eval_transform  = get_transforms(mode='eval')

    # Loading base training dataset
    base_train = torchvision.datasets.FashionMNIST(
        root="data",
        train=True,
        download=False,
        transform=None,  
    )

    n_total = len(base_train)                   
    n_val   = int(val_fraction * n_total)       
    n_train = n_total - n_val                   


    # Creating deterministic indices for train/val split
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=g)
    train_indices = indices[:n_train]
    val_indices   = indices[n_train:]

    # Actual training and validation dataset
    full_train = torchvision.datasets.FashionMNIST(
        root="data",
        train=True,
        download=False,
        transform=train_transform,
    )
    full_val = torchvision.datasets.FashionMNIST(
        root="data",
        train=True,
        download=False,
        transform=eval_transform,
    )

    train_data = torch.utils.data.Subset(full_train, train_indices)
    val_data   = torch.utils.data.Subset(full_val,   val_indices)

    test_data = torchvision.datasets.FashionMNIST(
        root="data",
        train=False,
        download=False,
        transform=eval_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader




def train_fashion_model(fashion_mnist, 
                        n_epochs, 
                        batch_size=64,
                        learning_rate=0.001,
                        USE_GPU=False,):
    """
    You can modify the contents of this function as needed, but DO NOT CHANGE the arguments,
    the function name, or return values, as this will be called during marking!
    (You can change the default values or add additional keyword arguments if needed.)
    """

    global train_loss_history, val_loss_history, val_acc_history
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []

    # 1) Choose device based on USE_GPU flag
    if USE_GPU and torch.backends.mps.is_available():
        device = torch.device("mps")        
    elif USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")       
    else:
        device = torch.device("cpu")

    # 2) Data
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=batch_size)

    # 3) Model, loss, optimizer
    model = Net().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    best_val_acc = 0.0
    best_state_dict = None

    # 4) Training loop over epochs
    for epoch in range(1, n_epochs + 1):
        train_loss = engine.train(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = engine.eval(model, val_loader, criterion, device)

        # For plotting
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(
            f"Epoch {epoch:02d}/{n_epochs} "
            f"- train_loss: {train_loss:.4f} "
            f"- val_loss: {val_loss:.4f} "
            f"- val_acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()

    # 5) Load best weights and evaluate on test set (for your own info)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        test_loss, test_acc = engine.eval(model, test_loader, criterion, device)
        print(f"Final test_loss: {test_loss:.4f}  - test_acc: {test_acc:.4f}")

    # Return the model's state_dict (weights) - DO NOT CHANGE THIS
    return model.state_dict()


def get_transforms(mode='train'):
    """
    Define any data augmentations or preprocessing here if needed.
    Only standard torchvision transforms are permitted (no lambda functions), please check that 
    these pass by running model_calls.py before submission. Transforms will be set to .eval()
    (deterministic) mode during evaluation, so avoid using stochastic transforms like RandomCrop
    or RandomHorizontalFlip unless they can be set to p=0 during eval.
    """
    mean = (0.5,)
    std  = (0.5,)

    if mode == 'train':
        return torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
    elif mode == 'eval':
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
    else:
        raise ValueError("mode must be 'train' or 'eval'")


def load_training_data():
    # Load FashionMNIST dataset
    # Do not change the dataset or its parameters
    print("Loading Fashion-MNIST dataset...")
    fashion_mnist = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=False,
    )
    # We load in data as the raw PIL images - recommended to have a look in visualise_dataset.py! 
    # To use them for training or inference, we need to transform them to tensors. 
    # We set this transform here, as well as any other data preprocessing or augmentation you 
    # wish to apply.
    fashion_mnist.transform = get_transforms(mode='train')
    return fashion_mnist


def main():
    # example usage
    # you could create a separate file that calls train_fashion_model with different parameters
    # or modify this as needed to add cross-validation, hyperparameter tuning, etc.
    fashion_mnist = load_training_data()

    # TODO: create data splits

    # TODO: implement hyperparameter search

    # Train model 
    # TODO: this may be done within a loop for hyperparameter search / cross-validation
    model_weights = train_fashion_model(
        fashion_mnist,
        n_epochs=25,          # train longer
        batch_size=32,
        learning_rate=1e-3,
        USE_GPU=True
    )


    # Save model weights
    # However you tune and evaluate your model, make sure to save the final weights 
    # to submission/model_weights.pth before submission!
    model_save_path = os.path.join('submission', 'model_weights.pth')
    torch.save(model_weights, f=model_save_path)


if __name__ == "__main__":
    main()