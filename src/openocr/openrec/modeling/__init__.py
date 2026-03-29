import copy

from .base_recognizer import BaseRecognizer

__all__ = ['build_model']


MODULES = {
    "BaseRecognizer": BaseRecognizer
}

def build_model(config):
    config = copy.deepcopy(config)
    rec_model = BaseRecognizer(config)
    return rec_model
