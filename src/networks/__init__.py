import sys

from .dcgan import dcgan
from .net import Net


class NetFactory(object):

    @staticmethod
    def create(*args, **kwargs):
        name = kwargs.pop('name')
        return getattr(sys.modules[__name__], name)(*args, **kwargs)
