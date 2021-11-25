import importlib
from typing import Union
from argparse import Namespace
import torch


def build_model(args: Namespace, device:torch.device) -> Union[torch.nn.Module]:
    cls = getattr(importlib.import_module('modules.models.{}'.format(args.model_module)), '{}'.format(args.model_name))
    model = cls(args, device)
    return model
