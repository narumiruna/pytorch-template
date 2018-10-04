import sys

from .net import Net


class NetworkFactory(object):

    @staticmethod
    def create(*args, **kwargs):
        name = kwargs.pop('name')
        return getattr(sys.modules[__name__], name)(*args, **kwargs)
