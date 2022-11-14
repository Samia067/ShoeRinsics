
from codes.model.UNetDecomposer import UNetDecomposer
from codes.model.UNetRenderer import UNetRenderer

models = {}
def register_model(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def get_model(name, **args):
    net = models[name](**args)
    return net


@register_model('decomposer')
def decomposer(weights_init=None, out_range='0, 1', output_last_ft=True):
    model = UNetDecomposer(out_range=out_range, output_last_ft=output_last_ft)
    if weights_init is not None:
        model.load_pretrained_model(weights_init)
    return model

@register_model('renderer')
def renderer(weights_init=None, out_range='0,1'):
    model = UNetRenderer(out_range=out_range)
    if weights_init is not None:
        model.load_pretrained_model(weights_init)
    return model


