from .base import BaseTransforms

import torchaudio.transforms as T


class DefaultTransforms(BaseTransforms):
    def __init__(self) -> None:
        super().__init__()

    def train_transform(self):
        return None

    def val_transform(self):
        return self.train_transform()

    def test_transform(self):
        return self.train_transform()
