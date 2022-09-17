import os
import argparse
import yaml
import torch
from yaml import Loader
import collections.abc


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


class Config():
    __dictpath__ = ''

    def __init__(self, d=None):
        d = d or {}
        for key, value in d.items():
            self_value = getattr(self, key)
            type_value = type(self_value) if type(self_value) is not type else self_value
            if isinstance(self_value, Config) or issubclass(type_value, Config):
                value = self_value.from_dict(value)
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        for key, value in d.items():
            if hasattr(cls, key):
                cls_value = getattr(cls, key)
                type_value = type(cls_value) if type(cls_value) is not type else cls_value
                if isinstance(cls_value, Config) or issubclass(type_value, Config):
                    value = cls_value.from_dict(value)
                setattr(cls, key, value)
        return cls

    @classmethod
    def from_file(cls, filepath, dictpath=None):
        with open(filepath, mode='r') as f:
            all_config = yaml.load(f, Loader=Loader)

        dictpath = dictpath or cls.__dictpath__
        my_config = all_config
        try:
            for key in dictpath.split('.'):
                if key != '':
                    my_config = my_config[key]
        except:
            my_config = None

        cls.from_dict(my_config)
        return cls

    @classmethod
    def to_dict(cls):
        d = dict(cls.__dict__)
        for key, value in cls.__dict__.items():
            if '__' in key and key not in ['__dictpath__', '__doc__']:
                d.pop(key, None)
            type_value = type(value) if type(value) is not type else value
            if isinstance(value, Config) or issubclass(type_value, Config):
                d[key] = value.to_dict()
        return d

    @classmethod
    def to_file(cls, filepath, dictpath=None, mode='merge_cls'):
        """to_file.
        Parameters
        ----------
        filepath :
            filepath
        dictpath :
            dictpath
        mode : str
            mode defines behaviour when file exists
            'new': create new one.
            'merge_cls': merge and prioritize current settings on cls
            'merge_file': merge and prioritize settings on file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        d = cls.to_dict()
        dictpath = dictpath or d['__dictpath__']
        my_config = d
        for key in reversed(dictpath.split('.')):
            if key != '':
                my_config = {key: my_config}

        config = {}
        if os.path.exists(filepath) and mode != 'new':
            with open(filepath, mode='r') as file:
                config = yaml.load(file, Loader=Loader)

        if mode == 'merge_file':
            recursive_update(my_config, config)
            config = my_config
        else:
            recursive_update(config, my_config)

        with open(filepath, mode='w') as file:
            yaml.dump(config, file)
