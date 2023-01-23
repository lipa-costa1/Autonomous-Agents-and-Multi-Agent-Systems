import numpy as np
from rlgym.utils import StateSetter
from rlgym.utils.common_values import CAR_MAX_SPEED, BALL_RADIUS, CAR_MAX_ANG_VEL, \
    BALL_MAX_SPEED
from rlgym.utils.math import rand_vec3
from rlgym.utils.state_setters import StateWrapper

from ._constants import LIM_X, LIM_Y, LIM_Z, PITCH_LIM, YAW_LIM, ROLL_LIM


class BetterRandom(StateSetter):
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        """
        Sets the following conditions:
            - Ball random position [uniform]
            - Ball random velocity with exponential distribution (higher velocity has less probability) [exponential]
            - Ball random angular velocity [triangular]
            - Car random position with usually 1 second away from ball (at max speed) [uniform]
                -> Fall back to random if out of field
            - Car random velocity [triangular]
            - Car random pitch and roll [triangular]
            - Car random yaw [uniform]
            - Car random angular velocity [triangular]
            - Car random boost amount [uniform]

        Args:
            state_wrapper (StateWrapper): Data class to permit the manipulation of environment variables.
        """
        state_wrapper.ball.set_pos(
            x=np.random.uniform(-LIM_X, LIM_X),
            y=np.random.uniform(-LIM_Y, LIM_Y),
            z=BALL_RADIUS,
        )

        # 99.9% chance of below ball max speed
        ball_speed = np.random.exponential(-BALL_MAX_SPEED / np.log(1 - 0.999))
        vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED))
        state_wrapper.ball.set_lin_vel(*vel)

        ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL + 0.5))
        state_wrapper.ball.set_ang_vel(*ang_vel)

        for car in state_wrapper.cars:
            # On average 1 second at max speed away from ball
            ball_dist = np.random.exponential(BALL_MAX_SPEED)
            ball_car = rand_vec3(ball_dist)
            car_pos = state_wrapper.ball.position + ball_car
            if abs(car_pos[0]) < LIM_X \
                    and abs(car_pos[1]) < LIM_Y \
                    and 0 < car_pos[2] < LIM_Z:
                car.set_pos(*car_pos)
            else:  # Fallback on fully random
                car.set_pos(
                    x=np.random.uniform(-LIM_X, LIM_X),
                    y=np.random.uniform(-LIM_Y, LIM_Y),
                    z=BALL_RADIUS,
                )

            vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
            car.set_lin_vel(*vel)

            car.set_rot(
                pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
            )

            ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
            car.set_ang_vel(*ang_vel)
            car.boost = np.random.uniform(0, 1)


class AasmaStateSetter(StateSetter):
    def __init__(
            self, *,
            random_prob: float = 1,
    ):
        """
        Define probability of each state setter

        Args:
            random_prob (float): probability of random setter
        """
        super().__init__()

        self.setters = [
            BetterRandom(),
        ]
        self.probs = np.array([random_prob])
        assert self.probs.sum() == 1, "Probabilities must sum to 1"

    def reset(self, state_wrapper: StateWrapper):
        """
        Sets state according to the probabilities of each state setter

        Args:
            state_wrapper (StateWrapper): Data class to permit the manipulation of environment variables.
        """
        i = np.random.choice(len(self.setters), p=self.probs)
        self.setters[i].reset(state_wrapper)