from collections import deque, namedtuple

import numpy as np

Experience = namedtuple(
    "Experience", ["state", "action", "reward", "done", "next_state"]
)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def append(self, experience: Experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        s, a, r, d, s2 = zip(*[self.buffer[idx] for idx in indices])
        return (
            np.array(s),
            np.array(a),
            np.array(r, dtype=np.float32),
            np.array(d, dtype=bool),
            np.array(s2),
        )

    def __len__(self):
        return len(self.buffer)
