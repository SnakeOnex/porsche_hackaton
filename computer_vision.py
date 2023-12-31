
# add your global imports here
import cv2
import time
import numpy as np
import pygame
import random
from pathlib import Path
import lane_detection
from center_line import CenterLinePredictor
from steering_alg import control


"""
We expect from you to implement a run_step() function that will return data required by the Driver to steer the car.
Our example uses RGB camera and OpenCV, but you can come up with any approach you wish
"""


class Navigator(object):

    def __init__(self, world):
        self.world = world
        self.lidar = world.lidar_manager
        self.camera_rgb = world.camera_manager

        self.frame_count = 0
        self.episode_folder = Path("data") / f"episode_{time.time():.0f}"
        self.image_folder = self.episode_folder / "images"
        self.steer_folder = self.episode_folder / "steer"
        self.image_folder.mkdir(parents=True)
        self.steer_folder.mkdir(parents=True)
        self.steering_vals_path = self.steer_folder / "steering_vals.txt"
        print("path: ", str(self.steering_vals_path))
        self.file = open(str(self.steering_vals_path), "w")
        self.detector = CenterLinePredictor()

    def run_step(self):
        """
        Method analyzing the sensors to obtain the data needed for car steering
        
        Returns:
            control_data: dict
        """

        control_data = dict()

        """
        The following lines will provide you with the image data from Camera Sensor
        - You can select position of the sensor manually in game mode before initiating the automatic control
        - pygame library provides a surface class that can be used to get 3-D image array to be later used in
            opencv
        - However the surface array has swapped axes, so it cannot be used directly in opencv
        """
        #print(self.frame_count)
        controls = self.world.player.get_control()
        throttle = controls.throttle
        steer = controls.steer
        brake = controls.brake
        #print(f"throttle={throttle}, steer={steer}, brake={brake}")
        #cv2.imshow("OpenCV camera view", self.camera_rgb.image)
        points = None
        poly = None
        try:
            im = self.detector.predict_to_image(self.camera_rgb.image)
            points, poly = self.detector.predict_to_points(self.camera_rgb.image)
            
            cv2.imshow("OpenCV camera view", im)
        except:
            pass
        # lane_detection.detect_edges(self.camera_rgb.image)
        if self.frame_count >= 50 and self.frame_count % 5 == 0:
            print("recording")
            image_path = self.image_folder / f"camera{self.frame_count:08d}.png"
            cv2.imwrite(str(image_path), self.camera_rgb.image)
            self.file.write(f"{steer}\n")
        cv2.waitKey(1)
        
        
        """
        Here, you can see exemplary data you can return from this step. It should help the driver
        fulfill the tasks
        """
        control_data["target_speed"] = random.randint(0, 50)
        control_data["curve"] = random.randint(-90, 90)
        control_data["p"] = points
        control_data["poly"] = poly

        self.frame_count += 1
        return control_data

