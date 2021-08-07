import os
import importlib

CRITERION_REGISTRY = dict()
CRITERION_DATACLASS_REGISTRY = dict()


def register_criterion(name: str, dataclass=None):
    r"""
    New criterion types can be added to OpenSpeech with the :func:`register_criterion` function decorator.

    For example::
        @register_criterion('label_smoothed_cross_entropy')
        class LabelSmoothedCrossEntropyLoss(nn.Module):
            (...)

    .. note:: All criterion must implement the :class:`cls.__name__` interface.

    Args:
        name (str): the name of the criterion
        dataclass (Optional, str): the dataclass of the criterion (default: None)
    """

    def register_criterion_cls(cls):
        if name in CRITERION_REGISTRY:
            raise ValueError(f"Cannot register duplicate criterion ({name})")

        CRITERION_REGISTRY[name] = cls

        cls.__dataclass = dataclass
        if dataclass is not None:
            if name in CRITERION_DATACLASS_REGISTRY:
                raise ValueError(f"Cannot register duplicate criterion ({name})")
            CRITERION_DATACLASS_REGISTRY[name] = dataclass

        return cls

    return register_criterion_cls


criterion_dir = os.path.dirname(__file__)
for file in os.listdir(criterion_dir):
    if os.path.isdir(os.path.join(criterion_dir, file)) and not file.startswith('__'):
        for subfile in os.listdir(os.path.join(criterion_dir, file)):
            path = os.path.join(criterion_dir, file, subfile)
            if subfile.endswith(".py"):
                python_file = subfile[: subfile.find(".py")] if subfile.endswith(".py") else subfile
                module = importlib.import_module(f"deepaudio.speaker.criterion.{file}.{python_file}")
        continue

    path = os.path.join(criterion_dir, file)
    if file.endswith(".py"):
        criterion_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(f"deepaudio.speaker.criterion.{criterion_name}")
