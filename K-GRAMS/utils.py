import matplotlib.pyplot as plt
import os

def make_directory(directoryPath):
    if not os.path.isdir(directoryPath):
        os.makedirs(directoryPath)
    return directoryPath


def plot_loss(train_loss, eval_loss, loss_function, experiment_name):
    plt.plot(train_loss, label = "Train Loss")
    plt.plot(eval_loss, label = "Test Loss")
    plt.legend()
    results_path = make_directory(os.path.join("../results", experiment_name))
    plt.savefig(os.path.join("../results", experiment_name, loss_function + ".eps"))
    plt.close()