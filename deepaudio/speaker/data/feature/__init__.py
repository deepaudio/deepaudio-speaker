import os
import importlib

AUDIO_FEATURE_TRANSFORM_REGISTRY = dict()
AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY = dict()


def register_audio_feature_transform(name: str, dataclass=None):
    r"""
    New dataset types can be added to OpenSpeech with the :func:`register_dataset` function decorator.

    For example::
        @register_audio_feature_transform("fbank", dataclass=FilterBankConfigs)
        class FilterBankFeatureTransform(object):
            (...)

    .. note:: All dataset must implement the :class:`cls.__name__` interface.

    Args:
        name (str): the name of the dataset
        dataclass (Optional, str): the dataclass of the dataset (default: None)
    """

    def register_audio_feature_transform_cls(cls):
        if name in AUDIO_FEATURE_TRANSFORM_REGISTRY:
            raise ValueError(f"Cannot register duplicate audio ({name})")

        AUDIO_FEATURE_TRANSFORM_REGISTRY[name] = cls

        if dataclass is not None:
            if name in AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY:
                raise ValueError(f"Cannot register duplicate dataclass ({name})")
            AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY[name] = dataclass

        return cls

    return register_audio_feature_transform_cls


data_dir = os.path.dirname(__file__)
for file in os.listdir(f"{data_dir}"):
    if os.path.isdir(f"{data_dir}/{file}") and not file.startswith('__'):
        path = f"{data_dir}/{file}"
        for module_file in os.listdir(path):
            path = os.path.join(path, module_file)
            if module_file.endswith(".py"):
                module_name = module_file[: module_file.find(".py")] if module_file.endswith(".py") else module_file
                module = importlib.import_module(f"deepaudio.speaker.data.feature.{file}.{module_name}")