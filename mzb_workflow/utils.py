from torchvision import models

from pathlib import Path

import torch
import torch.nn as nn

import numpy as np


def noneparse(value):
    """
    Helper function to parse None values from YAML files

    Parameters
    ----------
    value: string
        string to be parsed

    Returns
    -------
    value: string or None
        parsed string
    """

    if value.lower() == "none":
        return None

    return value


class cfg_to_arguments(object):
    """
    This class is used to convert a dictionary to an object and extend the argparser.
    In the __init__ method, we iterate over the dictionary and add each key as an attribute to the object.
    Input is a dictionary, output is an object, that mimicks the argparse object.

    Example:
    cfg = {'a': 1, 'b': 2}
    args = cfg_to_arguments(cfg)
    print(args.a) # 1
    print(args.b) # 2

    cfg can be from configs stored in YAML file, a JSON file, or a dictionary, whatever you prefer.
    """

    def __init__(self, args):
        """
        Parameters
        ----------
        args: dict
            dictionary of arguments
        """
        for key in args:
            setattr(self, key, args[key])

    def __str__(self):
        return self.__dict__.__str__()


def read_pretrained_model(architecture, n_class):
    """
    Helper script to load models compactly from pytorch model zoo and prepare them for Hummingbird finetuning

    Parameters
    ----------
    architecture: str
        name of the model to load
    n_class: int
        number of classes to finetune the model for

    Returns
    -------
    model : pytorch model
        model with last layer replaced with a linear layer with n_class outputs
    """

    architecture = architecture.lower()

    if architecture == "vgg":
        model = models.vgg16(pretrained=True)

        in_feat = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            in_features=in_feat, out_features=n_class, bias=True
        )

        for param in model.features.parameters():
            param.requires_grad = False

        for param in model.classifier.parameters():
            if np.any([a == 2 for a in param.shape]):
                pass
            else:
                param.requires_grad = False

    elif architecture == "resnet18":

        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=n_class, bias=True
        )

        # Freeze base feature extraction trunk:
        for param in model.parameters():
            param.requires_grad = True

        for param in model.fc.parameters():
            param.requires_grad = True

    elif architecture == "resnet50":

        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(
            in_features=model.fc.in_features, out_features=n_class, bias=True
        )

        # Freeze base feature extraction trunk:
        for param in model.parameters():
            param.requires_grad = True

        # for param in model.fc.parameters():
        #     param.requires_grad = True

    elif architecture == "densenet161":

        model = models.densenet161(pretrained=True)
        model.classifier = nn.Linear(
            in_features=model.classifier.in_features, out_features=n_class, bias=True
        )

        for param in model.parameters():
            param.requires_grad = True

        for param in model.classifier.parameters():
            param.requires_grad = True

    elif architecture == "mobilenet":

        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=n_class,
            bias=True,
        )

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier[1].parameters():
            param.requires_grad = True

    elif architecture == "efficientnet_b2":
        model = models.efficientnet_b2(pretrained=True)
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=n_class,
            bias=True,
        )

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier[1].parameters():
            param.requires_grad = True

    elif architecture == "efficientnet_b1":
        model = models.efficientnet_b1(pretrained=True, progress=False)
        model.classifier[1] = nn.Linear(
            in_features=model.classifier[1].in_features,
            out_features=n_class,
            bias=True,
        )

        for param in model.parameters():
            param.requires_grad = False

        for param in model.classifier[1].parameters():
            param.requires_grad = True

    elif architecture == "vit16":

        model = models.vit_b_16(pretrained=True)
        model.heads.head = nn.Linear(
            in_features=model.heads.head.in_features,
            out_features=n_class,
            bias=True,
        )

        for param in model.parameters():
            param.requires_grad = False

        for param in model.heads.head.parameters():
            param.requires_grad = True
    elif architecture == "convnext_small":
        model = models.vit_b_16(pretrained=True)

    else:

        raise OSError("Model not found")

    return model


def find_checkpoints(dirs=Path("lightning_logs"), version=None, log="val"):
    """
    Find the checkpoints for a given log

    Parameters
    ----------
    dirs: Path  (default: Path("lightning_logs"))
        path to the lightning_logs folder
    version: str (default: None)
        version of the log to use

    Returns
    -------
    chkp: str
        list of paths to checkpoints

    """

    if version:
        ch_sf = list(dirs.glob(f"{version}/checkpoints/*.ckpt"))
    else:  # pick last
        ch_sp = [a.parents[1] for a in dirs.glob("**/*.ckpt")]
        ch_sp.sort()
        ch_sf = list(ch_sp[-1].glob("**/*.ckpt"))

    chkp = [a for a in ch_sf if log in str(a.name)]

    return chkp
