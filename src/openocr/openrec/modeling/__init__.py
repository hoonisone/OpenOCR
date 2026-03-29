import copy

from .base_recognizer import BaseRecognizer

__all__ = ['build_model']


MODULES = {
    "BaseRecognizer": ".base_recognizer"
}

from importlib import import_module

def build_model(config):
    config = copy.deepcopy(config)
    module_name = config.pop('name')
    if module_name not in MODULES:
        raise ValueError(f'Unsupported model: {module_name}')
    module_str = MODULES[module_name]
    module = import_module(module_str, package=__package__)
    module_class = getattr(module, module_name)
    return module_class(config)
    # rec_model = BaseRecognizer(config)
    # return rec_model
