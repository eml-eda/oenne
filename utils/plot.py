import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from itertools import groupby
from torcheval.metrics import MulticlassConfusionMatrix

# plot learning curves from a history dataframe generated during training
def plot_learning_curves(history, w = 9):
    # fail gracefully if there is no history
    if history is None or len(history) == 0:
        print("Empty history, cannot plot")
        return

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot()
    epochs = range(len(history))
    loss = [h['loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    loss = np.convolve(np.pad(loss, (w-1)//2, mode='edge'), np.ones(w), 'valid') / w
    val_loss = np.convolve(np.pad(val_loss, (w-1)//2, mode='edge'), np.ones(w), 'valid') / w
    ax.plot(epochs, loss, color='green', label='Train')
    ax.plot(epochs, val_loss, color='orange', label='Val.')
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    plt.tight_layout()
    plt.show()

# similar to the previous function, but for a NAS loop (also plots the cost)
def plot_learning_curves_nas(history, w = 9):
    # fail gracefully if there is no history
    if history is None or len(history) == 0:
        print("Empty history, cannot plot")
        return

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot()
    epochs = range(len(history))
    loss = [h['loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    nas_loss = [h['nas_loss'] for h in history]
    cost = [h['cost'] for h in history]
    loss = np.convolve(np.pad(loss, (w-1)//2, mode='edge'), np.ones(w), 'valid') / w
    val_loss = np.convolve(np.pad(val_loss, (w-1)//2, mode='edge'), np.ones(w), 'valid') / w
    nas_loss = np.convolve(np.pad(nas_loss, (w-1)//2, mode='edge'), np.ones(w), 'valid') / w
    cost = np.convolve(np.pad(cost, (w-1)//2, mode='edge'), np.ones(w), 'valid') / w
    line1, = ax.plot(epochs, loss, color='green', label='Train Loss')
    line2, = ax.plot(epochs, val_loss, color='orange', label='Val. Loss')
    line3, = ax.plot(epochs, nas_loss, color='red', label='Tot. NAS Loss')
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    ax2 = ax.twinx()
    line4, = ax2.plot(epochs, cost, color='blue', label='DNN Cost', linestyle='dashed')
    ax2.set_ylabel('Cost')

    # Combine legends
    lines = [line1, line2, line4, line3]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels)

    plt.tight_layout()
    plt.show()


# computes the confusion matrix of model on eval_dl
# it does it by iterating over a dataloader and accumulating in a torcheval metric
def get_conf_matrix(model, eval_dl, device):
    conf = MulticlassConfusionMatrix(
        device=device, num_classes=len(CLASS_NAMES))
    model.eval()
    with torch.no_grad():
        for sample, target in eval_dl:
            sample, target = sample.to(device), target.to(device)
            output = model(sample)
            conf.update(output, target)
    return conf.compute().cpu().numpy()


# computes the confusion matrix of model on eval_dl using seaborn
def plot_conf_matrix(model, eval_dl, device):
    conf = get_conf_matrix(model, eval_dl, device)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()
    df_cm = pd.DataFrame(conf.astype(
        int), index=CLASS_NAMES, columns=CLASS_NAMES)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='OrRd', ax=ax)
    ax.set_title("Normalized confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    plt.show()


# find pareto optimal models from two paired arrays, where the first metric
# shall be minimized and the second maximized
def pareto_frontier(Xs, Ys):
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))])
    p_front = [myList[0]]
    for pair in myList[1:]:
        if pair[1] >= p_front[-1][1]:
            p_front.append(pair)
    p_front = np.array(p_front)
    return p_front[:, 0], p_front[:, 1]


# Plot models in the accuracy vs size plane, and highlight the Pareto frontier
def plot_pareto(size, accuracy, names):
    pareto_sizes, pareto_accuracies = pareto_frontier(size, accuracy)
    names = ["Seed",] + names
    plt.figure(figsize=(6, 6))
    # Plot the first point as a black diamond (seed)
    plt.scatter(size[0], accuracy[0], label='Seed',
                color='black', marker='D', s=100)
    # Plot the rest of the points as orange dots
    plt.scatter(size[1:], accuracy[1:], label='Optimized Models',
                color='orange', edgecolors='black', linewidths=1.5, s=100)
    # Plot the Pareto frontier
    plt.plot(pareto_sizes, pareto_accuracies,
             label='Pareto Frontier', color='black', linestyle='--')
    # Add names to the plot
    for i in range(1, len(size)):
        plt.text(size[i], accuracy[i], f' {names[i]}',
                 verticalalignment='top', horizontalalignment='left', fontsize=12)
    plt.xlabel('N. of Parameters')
    plt.ylabel('Accuracy')
    plt.title('Model Size vs. Accuracy with Pareto Frontier')
    plt.legend()
    plt.grid(True)
    plt.show()
