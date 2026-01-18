class SimpleSolution3:
    def __init__(self, params, gravity_magnitude=None, print_=False):
        self.params = params
        self.gravity_magnitude = gravity_magnitude
        self.print_ = print_

    def compute_action(self, observation):
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        theta, vtheta = observation[4], observation[5]
        leg1, leg2 = observation[6], observation[7]

        low_speed_height = self.params[0]
        low_speed = self.params[1]
        high_speed = self.params[2]
        x_gain = self.params[3]
        max_tilt = self.params[4]
        torque_gain = self.params[5]

        if y < low_speed_height:
            vy_target = -low_speed
        else:
            vy_target = -high_speed
        # if abs(x) > 1.0:
        #     vy_target = 0.0

        if vy < vy_target:
            thrust = 1  # main engine
        else:
            thrust = 0  # cut main engine

        target_theta = x_gain * x
        if target_theta < -max_tilt:
            target_theta = -max_tilt
        if target_theta > max_tilt:
            target_theta = max_tilt
        torque = -torque_gain * (target_theta - theta)
        if self.print_:
            print(f"x: {x:2f}\t vy: {vy:2f} \t Theta: {theta:2f} \t Target theta: {target_theta:2f}\t Action: {[thrust, torque]}")

        return [thrust, torque]