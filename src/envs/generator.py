import gymnasium as gym


class GymEnvGenerator(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(gym.make(*args, **kwargs))
