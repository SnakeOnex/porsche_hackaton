
# add your global imports here
import cv2
import numpy as np
import carla


"""
We expect from you to implement a run_step() function that will return vehicle control commands in form of 
carla.VehicleControl class (https://carla.readthedocs.io/en/latest/python_api/#carlavehiclecontrol)
"""


class Driver(object):

    def __init__(self, world=None):
        self.world = world
        self.vehicle = world.player if world else None
        pass

    def run_step(self, control_data: dict) -> carla.VehicleControl:
        """
        Method steering the car according to data provided

        Args:
            control_data: Data used to stear the car based on the Navigator function

        Returns:
            carla.VehicleControl
        """

        control = carla.VehicleControl()
        control.throttle = control_data["target_speed"] / 50  # Scalar value between [0.0,1.0]
        control.steer = control_data["curve"] / 90  # Scalar value between [-1.0, 1.0]; -1.0 max left, 1.0 max right
        control.brake = 0.0  # Scalar value between [0.0,1.0]
        control.hand_brake = False  # bool
        control.reverse = False  # bool
        control.manual_gear_shift = False  # bool - should be set to false
        control.gear = 1  # int - not used if manual_gear_shift == false
        return control
