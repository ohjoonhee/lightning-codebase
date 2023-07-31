from .base import BaseTransforms

import torchvision.transforms as T


class DefaultTransforms(BaseTransforms):
    def __init__(self) -> None:
        super().__init__()

    def train_transform(self):
        return T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def val_transform(self):
        return self.train_transform()

    def test_transform(self):
        return self.train_transform()
