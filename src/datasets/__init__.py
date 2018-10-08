import sys

from .mnist import mnist


class DatasetFactory(object):

    @staticmethod
    def create(*args, **kwargs):
        name = kwargs.pop('name')
        return getattr(sys.modules[__name__], name)(*args, **kwargs)
