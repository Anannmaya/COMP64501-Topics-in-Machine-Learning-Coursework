import matplotlib.pyplot as plt
from submission import fashion_training as ft


def main():
    # Run one training session to fill the histories
    fashion_mnist = ft.load_training_data()
    _ = ft.train_fashion_model(
        fashion_mnist,
        n_epochs=25,
        batch_size=32,
        learning_rate=1e-3,
        USE_GPU=True,
    )

    # histories now live inside ft.train_loss_history, etc.
    epochs = range(1, len(ft.train_loss_history) + 1)

    # Plot loss
    plt.figure()
    plt.plot(epochs, ft.train_loss_history, label="Train loss")
    plt.plot(epochs, ft.val_loss_history, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=200)

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, ft.val_acc_history, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("val_accuracy.png", dpi=200)


if __name__ == "__main__":
    main()
