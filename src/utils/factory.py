import mlflow


def log_params(params, prefix, sep='.'):
    for key, value in params.items():
        mlflow.log_param(prefix + sep + key, value)


def get_factory(obj):

    class Factory(object):

        @staticmethod
        def create(*args, **kwargs):
            name = kwargs.pop('name')
            log_params(kwargs, name)
            return getattr(obj, name)(*args, **kwargs)

    return Factory
