import importlib
from argparse import Namespace
import torch
from typing import Union


def build_components(args: Namespace) -> Union[torch.nn.Module]:
    module = importlib.import_module('modules.components.{}'.format(args.model_module))
    return module.build_components(args)