import math
import numpy as np
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

class MainController2:
    def __init__(self, flattened_params = None, gravity_magnitude = None, print_ = False):
        self.print_ = print_
        self.gravity_magnitude = gravity_magnitude
        if flattened_params is None:
            flattened_params = [-0.8, 0.0, 0.0,   # x position PID
                                0.5, 0.0, 1.0,   # y position PID
                                0.3, 0.0, 0.0,   # y velocity PID
                                -5.0, 0.0, 0.0]  # angle PID
                                # 5.0, 0.0, 1.0]  # angle rate PID
        
        x_kp, x_ki, x_kd = flattened_params[:3]
        y_kp, y_ki, y_kd = flattened_params[3:6]
        # vx_kp, vx_ki, vx_kd = flattened_params[6:9]
        vy_kp, vy_ki, vy_kd = flattened_params[6:9]
        angle_kp, angle_ki, angle_kd = flattened_params[9:12]
        
        # angle_rate_kp, angle_rate_ki, angle_rate_kd = flattened_params[15:18]

        self.x_position_controller = PIDController(kp=x_kp, ki=x_ki, kd=x_kd)
        self.y_position_controller = PIDController(kp=y_kp, ki=y_ki, kd=y_kd)
        # self.x_velocity_controller = PIDController(kp=vx_kp, ki=vx_ki, kd=vx_kd)
        self.y_velocity_controller = PIDController(kp=vy_kp, ki=vy_ki, kd=vy_kd)
        self.angle_controller = PIDController(kp=angle_kp, ki=angle_ki, kd=angle_kd)

        # self.angle_rate_controller = PIDController(kp=angle_rate_kp, ki=angle_rate_ki, kd=angle_rate_kd)
        self.assumed_mass = 0.1  # mass of the lander
        self.assumed_dt = 0.02  # assuming 60 FPS


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

        self.target_vy = self.y_position_controller.compute(setpoint=0, measurement=self.y, dt=self.assumed_dt)
        self.target_ay = self.y_velocity_controller.compute(setpoint=self.target_vy, measurement=self.vy, dt=self.assumed_dt)
        self.target_thrust = self.target_ay

        self.target_vx = self.x_position_controller.compute(setpoint=0, measurement=self.x, dt=self.assumed_dt)
        self.target_theta = np.clip(self.target_vx, -0.78, 0.78)
        self.target_torque = self.angle_controller.compute(setpoint=self.target_theta, measurement=self.theta, dt=self.assumed_dt)

        if self.leg1 and not self.leg2 and abs(self.vy) < 0.2:
            self.target_thrust = 0.0
            self.target_torque = 1.0
        elif self.leg2 and not self.leg1 and abs(self.vy) < 0.2:
            self.target_thrust = 0.0
            self.target_torque = -1.0
        elif self.leg1 and self.leg2:
            self.target_thrust = 0.0
            self.target_torque = 0.0
        if self.print_:
            print(f"self.target_theta: {self.target_theta:2f}\t self.target_ay: {self.target_ay:2f} \t Theta: {self.theta:2f} \t Target theta: {self.target_theta:2f}\t Action: {self.target_thrust:2f} {self.target_torque:2f}")

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
