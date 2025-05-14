import torch
from absl import app, flags
from ml_collections.config_flags import config_flags

import wandb
from prob_proto import pl_training
from prob_proto.losses import loss_factory
from prob_proto.models import model_factory

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    # "./configs/baseline/baseline_resnet50.py",
    "./configs/ppn_resnet50_birds.py",
    "Experiment Configuration",
)
flags.DEFINE_enum(
    "mode", "train", ["train", "debug", "test"], "Running mode: train or debug"
)
flags.DEFINE_enum("acc", "pl", ["torch", "pl"], "Accelerator: torch or lightning (pl)")
flags.DEFINE_string("checkpoint", None, "Checkpoint Path")

torch.set_float32_matmul_precision("medium")


def main(argv):
    cfg = FLAGS.config
    debug = FLAGS.mode == "debug"

    wandb.init(
        project=cfg.wandb.project,
        mode="offline" if debug else None,
        config=cfg.to_dict(),
        # tags=["tag1"],
    )

    torch.manual_seed(cfg.seed)
    model = model_factory.create_model(cfg.to_dict()["model"])
    loss_fn = loss_factory.create_loss(cfg.to_dict()["loss"])
    model.cuda()

    if FLAGS.checkpoint is not None:
        model.load_state_dict(torch.load(FLAGS.checkpoint)["model_state_dict"])

    if FLAGS.acc == "pl":
        pl_training.run_training(cfg, model, loss_fn, debug=debug)
    else:
        raise Exception("Only Pytorch Lightning is supported for now.")
    wandb.finish()


if __name__ == "__main__":
    app.run(main)
