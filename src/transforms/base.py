import abc

class BaseTransforms(abc.ABC):
    @abc.abstractmethod
    def train_transform(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def val_transform(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def test_transform(self):
        raise NotImplementedError