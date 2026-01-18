class SimpleSolution2:
    def __init__(self, gravity_magnitude=None, print_=False):
        self.gravity_magnitude = gravity_magnitude
        self.print_ = print_

    def compute_action(self, observation):
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        theta, vtheta = observation[4], observation[5]
        leg1, leg2 = observation[6], observation[7]

        if y < 0.4:
            vy_target = -0.05
        else:
            vy_target = -0.5

        if vy < vy_target:
            thrust = 1  # main engine
        else:
            thrust = 0  # cut main engine
        if abs(x) > 1.0:
            gain_x = 2.0
        else:
            gain_x = 1.0
        target_theta = gain_x * x
        if target_theta < -0.4:
            target_theta = -0.4
        if target_theta > 0.4:
            target_theta = 0.4
        torque = -3 * (target_theta - theta)
        if self.print_:
            print(f"x: {x:2f}\t vy: {vy:2f} \t Theta: {theta:2f} \t Target theta: {target_theta:2f}\t Action: {[thrust, torque]}")

        return [thrust, torque]