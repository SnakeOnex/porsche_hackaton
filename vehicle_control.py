
# add your global imports here
import cv2
import numpy as np
import carla
from pid import PID
import time
from steering_alg import control as ctr

"""
We expect from you to implement a run_step() function that will return vehicle control commands in form of 
carla.VehicleControl class (https://carla.readthedocs.io/en/latest/python_api/#carlavehiclecontrol)
"""


class Driver(object):

    def __init__(self, world=None):
        self.world = world
        self.vehicle = world.player if world else None
        self.throt = []
        self.brake = []
        self.gear = []
        self.speed = []
        self.bef = False
        self.last_time = time.perf_counter()
        self.yawrb = 0.0
        self.pid = PID()
        pass

    def run_step(self, control_data: dict,) -> carla.VehicleControl:
        """
        Method steering the car according to data provided

        Args:
            control_data: Data used to stear the car based on the Navigator function

        Returns:
            carla.VehicleControl
        """

        control = carla.VehicleControl()
        vel = np.array([self.world.player.get_velocity().x,self.world.player.get_velocity().y])
        speed = self.global_to_local(vel, self.world.imu_sensor.compass)
        if self.world.doors_are_open:
            tht = self.pid.step(20.0, speed[0], time.perf_counter() - self.last_time)
        else:
            tht = self.pid.step(12.0, speed[0], time.perf_counter() - self.last_time)
        #print(tht)
        if tht > 0.0:
            control.throttle = tht
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = -tht
       
        #control.throttle = tht# control_data["target_speed"] / 50  # Scalar value between [0.0,1.0]
        #control.steer = 0#control_data["curve"] / 90  # Scalar value between [-1.0, 1.0]; -1.0 max left, 1.0 max right
        #control.brake = 0.0  # Scalar value between [0.0,1.0]
        #print(self.world.imu_sensor.gyroscope)
        control.steer = ctr(control_data["p"],0,20,self.world.imu_sensor.gyroscope[2],self.yawrb,time.perf_counter() - self.last_time)
        control.hand_brake = False  # bool
        control.reverse = False  # bool
        control.manual_gear_shift = False  # bool - should be set to false
        control.gear = 1  # int - not used if manual_gear_shift == false
        self.throt.append(control.throttle)
        self.brake.append(control.brake)
        self.gear.append(control.gear)
        self.speed.append([self.world.player.get_velocity().x,self.world.player.get_velocity().y])
        if self.world.doors_are_open and not self.bef :
            np.save("throt.npy",np.array(self.throt))
            np.save("speed.npy",np.array(self.speed))
        #print(self.world.player.get_velocity())
        self.bef = self.world.doors_are_open
        self.yawrb = self.world.imu_sensor.gyroscope[2]
        self.last_time = time.perf_counter()
        return control


    def global_to_local(self,speed, ori) -> np.ndarray:
        """Convert points from global to local coordinates
        From standard (x,y) to (x',y') where x' is forward and y' is left

        Args:
            pos (np.ndarray): Position of the car in global coordinates
            ori (float): Orientation of the car in global coordinates - degrees!
            points (np.ndarray): Points to be converted

        Returns:
            np.ndarray: Converted points
        """
        car_heading = np.deg2rad(ori-90)
        R = np.array([[np.cos(car_heading), -np.sin(car_heading)],
                    [np.sin(car_heading), np.cos(car_heading)]])

        points = (R.T @ speed.T).T
        return points

