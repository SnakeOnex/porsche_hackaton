import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_sample(image, steer_gt, steer_pred=None, steering_arrow_length=40):
    steering_wheel_pos = image.shape[1] // 2, image.shape[2]
    plt.figure(figsize=(8, 8))
    plt.imshow(image.permute(1, 2, 0))
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[2], 0)
    title_str = f"Steer GT: {np.rad2deg(steer_gt):.2f}"
    if steer_pred is not None:
        title_str += f" Steer Pred: {np.rad2deg(steer_pred):.2f}"
    plt.arrow(steering_wheel_pos[0], steering_wheel_pos[1], steering_arrow_length * np.sin(
        steer_gt), -steering_arrow_length * np.cos(steer_gt), color='r', width=2, label='Steering angle ground truth')
    plt.arrow(steering_wheel_pos[0], steering_wheel_pos[1], steering_arrow_length * np.sin(steer_pred), -steering_arrow_length * np.cos(
        steer_pred), color='g', width=2, label='Steering angle prediction')
    plt.legend()
    plt.title(title_str)
    plt.show()

def val_to_cls(val):
    if val == 0:
        return "LEFT"
    elif val == 1:
        return "FORWARD"
    elif val == 2:
        return "RIGHT"

def plot_sample_disc(image, steer_gt, steer_pred=None, steering_arrow_length=40):
    steering_wheel_pos = image.shape[1] // 2, image.shape[2]
    # plt.figure(figsize=(8, 8))
    plt.imshow(image.permute(1, 2, 0))
    # plt.xlim(0, image.shape[1])
    # plt.ylim(image.shape[2], 0)
    title_str = f"Steer GT: {val_to_cls(steer_gt)}"
    if steer_pred is not None:
        title_str += f" Steer Pred: {val_to_cls(np.argmax(steer_pred))}"
        title_str += f" Steer Pred: {steer_pred}"
    # plt.arrow(steering_wheel_pos[0], steering_wheel_pos[1], steering_arrow_length * np.sin(steer_gt), -steering_arrow_length * np.cos(steer_gt), color='r', width=2, label='Steering angle ground truth')
    # plt.arrow(steering_wheel_pos[0], steering_wheel_pos[1], steering_arrow_length * np.sin(steer_pred), -steering_arrow_length * np.cos(steer_pred), color='g', width=2, label='Steering angle prediction')
    # plt.legend()
    plt.title(title_str)
