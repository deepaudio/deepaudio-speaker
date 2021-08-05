import os
import importlib

DATA_MODULE_REGISTRY = dict()


def register_data_module(name: str):
    """
    New data module types can be added to OpenSpeech with the :func:`register_data_module` function decorator.

    For example::
        @register_data_module('ksponspeech')
        class LightningKsponSpeechDataModule:
            (...)

    .. note:: All vocabs must implement the :class:`cls.__name__` interface.

    Args:
        name (str): the name of the vocab
    """

    def register_data_module_cls(cls):
        if name in DATA_MODULE_REGISTRY:
            raise ValueError(f"Cannot register duplicate data module ({name})")
        DATA_MODULE_REGISTRY[name] = cls
        return cls

    return register_data_module_cls


data_module_dir = os.path.dirname(__file__)
for file in os.listdir(data_module_dir):
    if os.path.isdir(os.path.join(data_module_dir, file)) and file != '__pycache__':
        for subfile in os.listdir(os.path.join(data_module_dir, file)):
            path = os.path.join(data_module_dir, file, subfile)
            if subfile.endswith(".py"):
                data_module_name = subfile[: subfile.find(".py")] if subfile.endswith(".py") else subfile
                module = importlib.import_module(f"deepaudio.speaker.datasets.{file}.{data_module_name}")
