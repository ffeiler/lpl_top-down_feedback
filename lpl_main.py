# Torch imports
# Python imports
import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torchvision.transforms as transforms
from pl_bolts.callbacks.printing import PrintTableMetricsCallback
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule
from pl_bolts.models.self_supervised.simclr import SimCLREvalDataTransform, SimCLRTrainDataTransform
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    imagenet_normalization,
    stl10_normalization,
)
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything

from callbacks.ssl_callbacks import SSLEvalCallback

# Custom imports
from models.module import LPL, NegSampleBaseline, SupervisedBaseline
from utils.utils import generate_descriptor, get_time_stamp


def add_hyperparameter_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    # Data params
    parser.add_argument("--dataset", type=str, choices=["cifar10", "imagenet2012", "stl10"], default="cifar10")
    parser.add_argument("--downsample_images", action="store_true")

    # Training hyperparams
    parser.add_argument("--max_epochs", type=int, default=800, help="number of total epochs to run")
    parser.add_argument("--max_steps", type=int, default=-1, help="max steps")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.5e-6)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--start_lr", type=float, default=0, help="initial warmup learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

    # Compute params
    parser.add_argument("--fast_dev_run", default=1, type=int)
    parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
    parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
    parser.add_argument("--num_workers", default=16, type=int, help="num of workers per GPU")
    parser.add_argument("--fp32", action="store_true")

    return parser


def cli_main():
    args = parse_args()

    # Set up random seeds for reproducibility
    seed_everything(args.random_seed)

    # Set up log directories (for experimental sanity) and tensorboard logger
    ouputdir = os.path.expanduser("../data/lpl")
    dataset = args.dataset
    if args.downsample_images:
        dataset = dataset + "_downsampled"
    experiment_descriptor = generate_descriptor(**args.__dict__)
    tensorboard_logger = pl_loggers.TensorBoardLogger(
        os.path.join(ouputdir, dataset), name=args.experiment_name, version=experiment_descriptor
    )

    # Create datamodule
    data_dir = os.path.join(os.path.expanduser("../data/datasets"), args.dataset)
    dm = None
    h = 0  # image size

    if args.dataset == "cifar10":
        dm = CIFAR10DataModule.from_argparse_args(args, data_dir=data_dir)
        normalization = cifar10_normalization()
        (c, h, w) = dm.size()

    elif args.dataset == "stl10":
        dm = STL10DataModule.from_argparse_args(args, data_dir=data_dir)
        normalization = stl10_normalization()
        if args.train_with_supervision:
            dm.train_dataloader = dm.train_dataloader_labeled
        else:
            dm.train_dataloader = dm.train_dataloader
        dm.val_dataloader = dm.train_dataloader_labeled
        (c, h, w) = dm.size()
        if args.downsample_images:
            h = 32

    elif args.dataset == "imagenet2012":
        dm = ImagenetDataModule.from_argparse_args(args, data_dir=data_dir, image_size=196)
        normalization = imagenet_normalization()
        (c, h, w) = dm.size()

    dm.train_transforms = SimCLRTrainDataTransform(h, normalize=normalization)
    dm.val_transforms = SimCLREvalDataTransform(h, normalize=normalization)
    dm.test_transforms = SimCLREvalDataTransform(h, normalize=normalization)

    if args.downsample_images:
        dm.train_transforms = transforms.Compose([transforms.Resize(32), dm.train_transforms])
        dm.val_transforms = transforms.Compose([transforms.Resize(32), dm.val_transforms])
        dm.test_transforms = transforms.Compose([transforms.Resize(32), dm.test_transforms])

    args.num_classes = dm.num_classes
    dm.prepare_data()
    dm.setup()

    if args.train_with_supervision:
        model = SupervisedBaseline(**args.__dict__)
    elif args.use_negative_samples:
        model = NegSampleBaseline(**args.__dict__)
    else:
        model = LPL(**args.__dict__)

    # callbacks
    printing = PrintTableMetricsCallback()

    if args.gpus > 1:
        print(
            "Online evaluation with multi-gpu training currently not supported, use notebooks for post-training evaluation instead"
        )
        callbacks = []
    else:
        repr_eval = SSLEvalCallback(num_epochs=20, test_dataloader=dm.test_dataloader())
        callbacks = [repr_eval]
    if args.verbose_printing:
        callbacks.append(printing)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        max_steps=None if args.max_steps == -1 else args.max_steps,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        accelerator="ddp" if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        callbacks=callbacks,
        logger=tensorboard_logger,
        profiler="simple",
    )
    trainer.fit(model, dm)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--random_seed", type=int, default=24)
    parser.add_argument("--experiment_name", type=str, default=get_time_stamp())
    parser.add_argument("--resume_from_checkpoint", action="store_true")

    parser.add_argument("--verbose_printing", action="store_true")

    # Model options
    parser.add_argument("--train_with_supervision", action="store_true")
    parser.add_argument("--use_negative_samples", action="store_true")
    parser.add_argument("--train_end_to_end", action="store_true")

    parser = add_hyperparameter_args(parser)
    parser = LPL.add_model_specific_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    cli_main()
