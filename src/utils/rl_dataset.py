from typing import Iterator, Tuple
from .replay_buffer import ReplayBuffer

from torch.utils.data import IterableDataset


class RLDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        super().__init__()

        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        state, action, reward, done, next_state = self.buffer.sample(self.sample_size)
        for i in range(len(done)):
            yield state[i], action[i], reward[i], done[i], next_state[i]
