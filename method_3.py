class SimpleSolution2:
    def __init__(self):
        pass

    def compute_action(self, observation):
        x, y = observation[0], observation[1]
        vx, vy = observation[2], observation[3]
        theta, vtheta = observation[4], observation[5]
        leg1, leg2 = observation[6], observation[7]

        if y < 0.5:
            vy_target = -0.1
        else:
            vy_target = -0.4

        if vy < vy_target:
            thrust = 1  # main engine
        else:
            thrust = 0  # cut main engine
        
        target_theta = 0.8 * x
        if target_theta < -0.4:
            target_theta = -0.4
        if target_theta > 0.4:
            target_theta = 0.4
        torque = -5 * (target_theta - theta)

        print(f"x: {x:2f}\t vy: {vy:2f} \t Theta: {theta:2f} \t Action: {[thrust, torque]}")

        return [thrust, torque]