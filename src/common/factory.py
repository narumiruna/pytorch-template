import inspect


class Factory(object):

    def __init__(self, module, key):
        self.module = module
        self.key = key

    def create(self, config, *args):
        if isinstance(config, dict):
            data = config[self.key]
            name = data['name']

            obj = getattr(self.module, name)

            if inspect.isclass(obj):
                varnames = obj.__init__.__code__.co_varnames
            elif inspect.isfunction(obj):
                varnames = obj.__code__.co_varnames

            # remove unexpected keyword argument
            kwargs = {k: v for k, v in data.items() if k in varnames}

            return obj(*args, **kwargs)
        else:
            raise TypeError
