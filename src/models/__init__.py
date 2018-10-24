import sys

from .lenet import LeNet


class ModelFactory(object):

    @staticmethod
    def create(*args, **kwargs):
        name = kwargs.pop('name')
        return getattr(sys.modules[__name__], name)(*args, **kwargs)
