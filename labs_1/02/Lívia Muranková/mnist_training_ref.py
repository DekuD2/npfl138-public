#!/usr/bin/env python3
import argparse

import torch
import torchmetrics

import npfl138
npfl138.require_version("2526.2")
from npfl138.datasets.mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--decay", default=None, choices=["linear", "exponential", "cosine"], help="Decay type")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=None, type=float, help="Nesterov momentum to use in SGD.")
parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam"], help="Optimizer to use.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


if __name__ == "__recodex_runner__":
    import os

    def load_input(runner, input_path):
        if runner.args.judge: os.replace("recodex_internal_mnist.npz", "mnist.npz")
        args = runner.parse_args_with_entrypoint_defaults(parser, runner.line(input_path), recodex=True)
        runner.torch_set_random_seed(args.seed)
        return [args], {}

    def judge(runner, result, gold):
        return runner.compare(result, gold, epsilon=1e-4)


class Dataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # a torch.Tensor with torch.uint8 values in [0, 255] range
        image = image.to(torch.float32) / 255  # image converted to float32 and rescaled to [0, 1]
        label = example["label"]  # a torch.Tensor with a single integer representing the label
        return image, label  # return an (input, target) pair


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads, args.recodex)
    npfl138.global_keras_initializers()

    # Load the data and create dataloaders.
    mnist = MNIST()

    train = torch.utils.data.DataLoader(Dataset(mnist.train), batch_size=args.batch_size, shuffle=True)
    dev = torch.utils.data.DataLoader(Dataset(mnist.dev), batch_size=args.batch_size)

    # Create the model.
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(MNIST.C * MNIST.H * MNIST.W, args.hidden_layer_size),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden_layer_size, MNIST.LABELS),
    )

    # Wrap the model in the TrainableModule.
    model = npfl138.TrainableModule(model)

    # TODO: Use the required `args.optimizer` (either `SGD` or `Adam`) with
    # the given `args.learning_rate`.
    # - For `SGD`, if `args.momentum` is specified, use Nesterov momentum.
    # - If `args.decay` is set, then also create a LR scheduler (otherwise, pass `None`).
    #   The scheduler should decay the learning rate from the initial `args.learning_rate`
    #   to the final `args.learning_rate_final`. The `scheduler.step()` is called after
    #   each batch, so the number of scheduler iterations is the number of batches in all
    #   training epochs (note that `len(train)` is the number of batches in one epoch).
    #   - for `linear`, use `torch.optim.lr_scheduler.LinearLR` and set `start_factor`,
    #     `end_factor`, and `total_iters` appropriately;
    #   - for `exponential`, use `torch.optim.lr_scheduler.ExponentialLR` and set `gamma`
    #     appropriately (be careful to compute it using float64 to avoid precision issues);
    #   - for `cosine`, use `torch.optim.lr_scheduler.CosineAnnealingLR` and set `T_max`
    #     and `eta_min` appropriately.
    #   In all cases, you should reach `args.learning_rate_final` just after the training.
    #
    #   If a learning rate schedule is used, the `TrainableModule` automatically logs the
    #   learning rate to the console and to TensorBoard. Additionally, you can find out
    #   the next learning rate to be used by printing `model.scheduler.get_last_lr()[0]`.
    #   Therefore, after the training, this value should be `args.learning_rate_final`.
    ##...
    optimizer_class = getattr(torch.optim, args.optimizer)
    optimizer_args = {"lr": args.learning_rate}
    if args.momentum:
        optimizer_args["momentum"] = args.momentum
        optimizer_args["nesterov"] = True
    optimizer = optimizer_class(model.parameters(), **optimizer_args)
    if args.decay:
        batches = args.epochs * len(train)
        if args.decay == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 1.0, args.learning_rate_final / args.learning_rate, batches)
        elif args.decay == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, (args.learning_rate_final / args.learning_rate) ** (1 / batches))
        elif args.decay == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, batches, eta_min=args.learning_rate_final)
    else:
        scheduler = None

    model.configure(
        ##optimizer=...,
        ##scheduler=...,
        # Solution
        optimizer=optimizer,
        scheduler=scheduler,
        # Stop
        loss=torch.nn.CrossEntropyLoss(),
        metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=MNIST.LABELS)},
        logdir=npfl138.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args)),
    )

    # Train the model.
    logs = model.fit(train, dev=dev, epochs=args.epochs)

    if args.decay:
        print(f"Next learning rate to be used: {model.scheduler.get_last_lr()[0]:g}")

    # Return development metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items() if metric.startswith("dev:")}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)

