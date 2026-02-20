import os
import json
import pathlib as pl

from cifar10_net.model import load_pretrained as _model_load_pretrained


def get_wandb_key():
    json_file = str(pl.PurePath("..", "wandb_key.json"))
    if os.path.isfile(json_file):
        with open(json_file, "r") as f:
            return json.load(f)
    elif "WANDB_KEY" in os.environ:
        return os.environ["WANDB_KEY"]


def load_pretrained(model, ckpt_path):
    """Backward-compatible wrapper around cifar10_net.model.load_pretrained."""
    pretrained_model = _model_load_pretrained(ckpt_path, device="cpu")
    model.load_state_dict(pretrained_model.state_dict())
    return model


def args_to_flat_dict(args):
    args_dict = vars(args.as_flat())
    for key in args_dict.keys():
        if args_dict[key] is None:
            args_dict[key] = "None"
    return args_dict
