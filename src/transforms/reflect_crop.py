import torchvision.transforms as T


from .base import BaseTransforms


class ReflectCropTransforms(BaseTransforms):
    def __init__(self, padding_size: int = 4) -> None:
        super().__init__()
        self.padding_size = padding_size

    def train_transform(self):
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomCrop(
                    (32, 32), padding=self.padding_size, padding_mode="reflect"
                ),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def val_transform(self):
        return T.Compose(
            [
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def test_transform(self):
        return self.val_transform()
