
from ptsemseg.loader import get_loader
from ptsemseg.augmentations import get_composed_augmentations, key2aug

def data_loader(config):
  return get_loader(config)

