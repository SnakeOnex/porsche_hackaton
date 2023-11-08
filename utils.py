import numpy as np
import matplotlib.pyplot as plt

def plot_sample(image, steer_gt, steer_pred=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(image.permute(1, 2, 0))
    title_str = f"Steer GT: {np.rad2deg(steer_gt):.2f}"
    if steer_pred is not None:
        title_str += f" Steer Pred: {np.rad2deg(steer_pred):.2f}"
    plt.title(title_str)
    plt.show()

