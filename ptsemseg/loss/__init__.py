
import functools
from ptsemseg.loss.loss import (
    cross_entropy2d,
    bootstrapped_cross_entropy2d,
    multi_scale_cross_entropy2d,
)


key2loss = {
    "cross_entropy": cross_entropy2d,
    "bootstrapped_cross_entropy": bootstrapped_cross_entropy2d,
    "multi_scale_cross_entropy": multi_scale_cross_entropy2d,
}


def get_loss_function(args):
    loss_name = args.loss
    if loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(loss_name))

    return functools.partial(key2loss[loss_name], size_average=args.size_average)

