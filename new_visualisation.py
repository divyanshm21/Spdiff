import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Change this to the correct path if needed

LOGS_DIR = "./experiments/c2dl/2025-04-07-15-37-47run"  # <-- adjust this!
EPOCH = 30  # or any epoch you have data for

# Load saved prediction and label data
p_pred = torch.load(os.path.join(LOGS_DIR, f"{EPOCH}_p_pred.pth")).cpu().numpy()
labels = torch.load(os.path.join(LOGS_DIR, f"{EPOCH}_labels.pth")).cpu().numpy()

# Visualize some pedestrian trajectories
def plot_trajectories(pred, gt, num_peds=5):
    plt.figure(figsize=(8, 8))
    for i in range(min(num_peds, pred.shape[1])):
        plt.plot(gt[:, i, 0], gt[:, i, 1], 'k--', label=f"GT {i}")
        plt.plot(pred[:, i, 0], pred[:, i, 1], label=f"Pred {i}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Predicted vs Ground Truth (Epoch {EPOCH})")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)
    plt.show()

plot_trajectories(p_pred, labels)
plt.savefig(f"{LOGS_DIR}/trajectory_plot_epoch_{EPOCH}.png")
