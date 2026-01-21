import math
import numpy as np
from methods.base_controller import BaseController

MODULE_CONFIG = {
    'class_name': 'CascadedPID',
    'params_file': 'method_2.json',
    'weights_file': None,
    'cma_num_params': 15,
}


class PIDController:
    def __init__(self, kp, ki, kd, integral_limit=10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0
        self.integral_limit = integral_limit

    def compute(self, setpoint, measurement, dt):
        error = setpoint - measurement
        self.integral += error * dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        derivative = (error - self.previous_error) / dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.previous_error = error
        return output


class CascadedPID(BaseController):
    # Intuition: Think of the lander as a drone. A Typical drone controller uses a cascaded PID 
    # controller to generate 3d thrust and torque commands from position setpoints.
    def __init__(self, gravity_magnitude, params, weights):
        super().__init__(params=params, weights=weights, gravity_magnitude=gravity_magnitude)
        
        if self.params is None:
            self.params = [0.1, 0.0, 0.0,
                            0.5, 0.0, 0.0,
                            0.0, 0.0, 0.0,
                            7.0, 0.0, 0.0,
                            0.0, 0.0, 0.0]
        
        x_kp, x_ki, x_kd = self.params[:3]
        y_kp, y_ki, y_kd = self.params[3:6]
        vx_kp, vx_ki, vx_kd = self.params[6:9]
        vy_kp, vy_ki, vy_kd = self.params[9:12]
        angle_kp, angle_ki, angle_kd = self.params[12:15]

        self.x_position_controller = PIDController(kp=x_kp, ki=x_ki, kd=x_kd)
        self.y_position_controller = PIDController(kp=y_kp, ki=y_ki, kd=y_kd)
        self.x_velocity_controller = PIDController(kp=vx_kp, ki=vx_ki, kd=vx_kd)
        self.y_velocity_controller = PIDController(kp=vy_kp, ki=vy_ki, kd=vy_kd)
        self.attitude_controller = PIDController(kp=angle_kp, ki=angle_ki, kd=angle_kd)

        self.assumed_mass = 0.1
        self.assumed_dt = 0.02

        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.theta = 0.0
        self.vtheta = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_ax = 0.0
        self.target_ay = 0.0
        self.target_theta = 0.0
        self.target_vtheta = 0.0
        self.target_thrust = 0.0
        self.target_torque = 0.0

        self.history = {
            "t": [],
            "x": [], "y": [],
            "vx": [], "vy": [],
            "theta": [], "vtheta": [],
            "action": [],
            "target_vx": [], "target_vy": [],
            "target_ax": [], "target_ay": [],
            "target_theta": [], "target_vtheta": [],
            "target_thrust": [], "target_torque": []
        }

    def compute_action(self, observation):
        self.x, self.y = observation[0], observation[1]
        self.vx, self.vy = observation[2], observation[3]
        self.theta, self.vtheta = observation[4], (observation[5] * 2.5)
        self.leg1, self.leg2 = observation[6], observation[7]

        self.target_vx = self.x_position_controller.compute(setpoint=0, measurement=self.x, dt=self.assumed_dt)
        self.target_ax = self.x_velocity_controller.compute(setpoint=self.target_vx, measurement=self.vx, dt=self.assumed_dt)

        self.target_vy = self.y_position_controller.compute(setpoint=0, measurement=self.y, dt=self.assumed_dt)
        self.target_ay = self.y_velocity_controller.compute(setpoint=self.target_vy, measurement=self.vy, dt=self.assumed_dt)

        self.target_theta = math.atan2(self.target_ay + self.gravity_magnitude, self.target_ax)
        self.target_theta -= math.pi / 2
        self.target_theta = np.clip(self.target_theta, -0.78, 0.78)

        self.target_vtheta = self.attitude_controller.compute(setpoint=self.target_theta, measurement=self.theta, dt=self.assumed_dt)
        self.target_torque = self.target_vtheta
        self.target_thrust = self.assumed_mass * math.hypot(self.target_ax, self.target_ay + self.gravity_magnitude)

        self.store_variables()
        return [self.target_thrust, self.target_torque]

    def store_variables(self):
        self.history["t"].append(len(self.history["t"]) * self.assumed_dt)
        self.history["x"].append(self.x)
        self.history["y"].append(self.y)
        self.history["vx"].append(self.vx)
        self.history["vy"].append(self.vy)
        self.history["target_vx"].append(self.target_vx)
        self.history["target_vy"].append(self.target_vy)
        self.history["target_ax"].append(self.target_ax)
        self.history["target_ay"].append(self.target_ay)
        self.history["theta"].append(self.theta)
        self.history["vtheta"].append(self.vtheta)
        self.history["target_theta"].append(self.target_theta)
        self.history["target_vtheta"].append(self.target_vtheta)
        self.history["target_thrust"].append(self.target_thrust)
        self.history["target_torque"].append(self.target_torque)
        self.history["action"].append([self.target_thrust, self.target_torque])

    def get_history(self):
        return self.history
