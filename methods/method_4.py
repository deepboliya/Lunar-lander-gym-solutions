from methods.base_controller import BaseController

MODULE_CONFIG = {
    'class_name': 'BasicController2',
    'params_file': None,
    'weights_file': None,
}


class BasicController2(BaseController):
    # Intuition: Similar to BasicController in method 1, but decrease the vertical speed when close to the ground
    # and add a small component to the target theta based on horizontal velocity so that it doesn't oscillate as much.
    # Target theta is now sort of a result of a PD controller on x position.
    # main thrust could have also been proportional to error in vertical speed but this works fine, so didn't change it.
    def __init__(self, gravity_magnitude, params, weights):
        super().__init__(gravity_magnitude=gravity_magnitude, params=params, weights=weights)

    def compute_action(self, observation):
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        theta, vtheta = observation[4], observation[5]
        leg1, leg2 = observation[6], observation[7]

        low_speed_height = 0.5
        low_speed = 0.1
        high_speed = 0.3
        x_gain = 1.0
        max_tilt = 0.5
        torque_gain = 5.0
        x_diff = 0.2

        if y < low_speed_height:
            vy_target = -low_speed
        else:
            vy_target = -high_speed

        if vy < vy_target:
            thrust = 1
        else:
            thrust = 0

        target_theta = x_gain * x + x_diff * vx

        if target_theta < -max_tilt:
            target_theta = -max_tilt
        if target_theta > max_tilt:
            target_theta = max_tilt
            
        torque = -torque_gain * (target_theta - theta)

        return [thrust, torque]
