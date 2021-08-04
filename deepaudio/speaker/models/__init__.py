import importlib
import os

from .speaker_model import SpeakerModel


MODEL_REGISTRY = dict()
MODEL_DATACLASS_REGISTRY = dict()


def register_model(name: str, dataclass=None):
    r"""
    New model types can be added to OpenSpeech with the :func:`register_model` function decorator.

    For example::
        @register_model('conformer_lstm')
        class ConformerLSTMModel(OpenspeechModel):
            (...)

    .. note:: All models must implement the :class:`cls.__name__` interface.

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Cannot register duplicate model ({name})")
        if not issubclass(cls, SpeakerModel):
            raise ValueError(f"Model ({name}: {cls.__name__}) must extend OpenspeechModel")

        MODEL_REGISTRY[name] = cls

        cls.__dataclass = dataclass
        if dataclass is not None:
            if name in MODEL_DATACLASS_REGISTRY:
                raise ValueError(f"Cannot register duplicate model ({name})")
            MODEL_DATACLASS_REGISTRY[name] = dataclass

        return cls

    return register_model_cls


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    if os.path.isdir(os.path.join(models_dir, file)) and not file.startswith('__'):
        for subfile in os.listdir(os.path.join(models_dir, file)):
            path = os.path.join(models_dir, file, subfile)
            if subfile.endswith(".py"):
                python_file = subfile[: subfile.find(".py")] if subfile.endswith(".py") else subfile
                module = importlib.import_module(f"deepMM.speaker.models.{file}.{python_file}")
        continue

    path = os.path.join(models_dir, file)
    if file.endswith(".py"):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(f"deepMM.speaker.models.{model_name}")