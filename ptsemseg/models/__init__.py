
from ptsemseg.models.fcn import fcn8s, fcn16s, fcn32s
from ptsemseg.models.segnet import segnet
from ptsemseg.models.unet import unet
from ptsemseg.models.pspnet import pspnet
from ptsemseg.models.icnet import icnet
from ptsemseg.models.linknet import linknet
from ptsemseg.models.frrn import frrn

from ptsemseg.models.stem import VGG16, resnet18

model_list = {
            "fcn32s": fcn32s,
            "fcn8s": fcn8s,
            "fcn16s": fcn16s,
            "unet": unet,
            "segnet": segnet,
            "pspnet": pspnet,
            "icnet": icnet,
            "icnetBN": icnet,
            "linknet": linknet,
            "frrnA": frrn,
            "frrnB": frrn,
            }

stem_list = {
    'resnet18': resnet18,
    'vgg16': VGG16
    }

def _get_model_instance(name):
    try:
        return model_list[name]
    except:
        raise ("Model {} not available".format(name))

def get_model(args, **kwargs):
    if 'vgg16' in kwargs:
        stem = stem_list['vgg16']
    elif 'resnet18' in kwargs:
        stem = stem_list['resnet18']
    else:
        stem = None
    if stem is None:
        raise ("stem not available")

    name = args.model
    seg_model = _get_model_instance(name)
    model = None
    if name in ["fcn32s", "fcn16s", "fcn8s"]:
        model = seg_model(stem, n_classes=args.n_classes, learned_billinear=args.learned_billinear, **kwargs)
        #vgg16 = models.vgg16(pretrained=True)
        #model.init_vgg16_params(vgg16)

    #elif name in ["frrnA", "frrnB"]:
    #    model = seg_model(n_classes, **param_dict)
    #elif name == "segnet":
    #    model = model(n_classes=n_classes, **param_dict)
    #    vgg16 = models.vgg16(pretrained=True)
    #    model.init_vgg16_params(vgg16)

    #elif name == "unet":
    #    model = model(n_classes=n_classes, **param_dict)

    #elif name == "pspnet":
    #    model = model(n_classes=n_classes, **param_dict)

    #elif name == "icnet":
    #    model = model(n_classes=n_classes, **param_dict)

    #elif name == "icnetBN":
    #    model = model(n_classes=n_classes, **param_dict)

    #else:
    #    model = model(n_classes=n_classes, **param_dict)

    if model is None:
        print("model not support")

    return model


