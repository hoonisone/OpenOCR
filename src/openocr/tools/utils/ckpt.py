import os

import torch

from openocr.tools.utils.logging import get_logger
from pathlib import Path

def save_ckpt(
    model,
    cfg,
    optimizer,
    lr_scheduler,
    epoch,
    global_step,
    metrics,
    is_best=False,
    logger=None,
    prefix=None,
):
    """
    Saving checkpoints

    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
    """

    weight_dir = cfg["Global"]["weights_dir"]
    weight_dir = Path(weight_dir).resolve()
    if not weight_dir.exists():
        os.makedirs(weight_dir, exist_ok=True)

    weight_dir = weight_dir.as_posix()
    if logger is None:
        logger = get_logger()
    if prefix is None:
        if is_best:
            save_path = os.path.join(weight_dir, "best.pth")
        else:
            save_path = os.path.join(weight_dir, "latest.pth")
    else:
        save_path = os.path.join(weight_dir, prefix + ".pth")


    state_dict = model.module.state_dict() if cfg["Global"]["distributed"] else model.state_dict()
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": state_dict,
        "optimizer": None if is_best else optimizer.state_dict(),
        "scheduler": None if is_best else lr_scheduler.state_dict(),
        "config": cfg,
        "metrics": metrics,
    }
    torch.save(state, save_path)
    logger.info(f"save ckpt to {save_path}")


def load_ckpt(model, cfg, optimizer=None, lr_scheduler=None, logger=None):
    """
    Resume from saved checkpoints
    :param checkpoint_path: Checkpoint path to be resumed
    """
    if logger is None:
        logger = get_logger()
    checkpoints = cfg["Global"].get("checkpoints")
    pretrained_model = cfg["Global"].get("pretrained_model")



    status = {}
    if checkpoints and os.path.exists(checkpoints):
        checkpoint = torch.load(checkpoints, map_location=torch.device("cpu"), weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(checkpoint["scheduler"])
        logger.info(f"resume from checkpoint {checkpoints} (epoch {checkpoint['epoch']})")

        status["global_step"] = checkpoint["global_step"]
        status["epoch"] = checkpoint["epoch"] + 1
        status["metrics"] = checkpoint["metrics"]
    elif pretrained_model and os.path.exists(pretrained_model):
        load_pretrained_params(model, pretrained_model, logger)
        logger.info(f"finetune from checkpoint {pretrained_model}")
    else:
        logger.info("train from scratch")
    return status


def load_pretrained_params(model, pretrained_model, logger):
    pretrained_model = Path(pretrained_model).resolve().as_posix()
    if pretrained_model.endswith(".safetensors"):
        from safetensors.torch import load_file
        logger.info(f"Loading weights from safetensors: {pretrained_model}")
        checkpoint = load_file(pretrained_model)
    else:
        logger.info(f"Loading weights using torch.load: {pretrained_model}")
        checkpoint = torch.load(pretrained_model, map_location=torch.device("cpu"), weights_only=False)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model_keys = model.state_dict().keys()
    for name in model_keys:
        if name not in state_dict:
            logger.info(f"{name} is not in pretrained model")

