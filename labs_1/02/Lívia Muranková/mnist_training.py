#c7a710d2-ba2b-428c-a5d1-d6ea22545173
#!/usr/bin/env python3
import argparse

import torch
import torchmetrics
import numpy as np

import npfl138
npfl138.require_version("2526.2")
from npfl138.datasets.mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--decay", default="exponential", choices=["linear", "exponential", "cosine"], help="Decay type")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer_size", default=128, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=0.0001, type=float, help="Final learning rate.")
parser.add_argument("--momentum", default=None, type=float, help="Nesterov momentum to use in SGD.")
parser.add_argument("--optimizer", default="Adam", choices=["SGD", "Adam"], help="Optimizer to use.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Dataset(npfl138.TransformedDataset):
    def transform(self, example):
        image = example["image"]  # a torch.Tensor with torch.uint8 values in [0, 255] range
        image = image.to(torch.float32) / 255  # image converted to float32 and rescaled to [0, 1]
        label = example["label"]  # a torch.Tensor with a single integer representing the label
        return image, label  # return an (input, target) pair

# Stochastic Gradient Descent optimizer
class SGD:
    def __init__(self, parameters, learning_rate, momentum):
        # objekt triedy SGD ma v sebe ulozene parametre (vahy a gradient), rychlost ucenia a momentum
        # vytvorit zoznam objektov parametrov - aby som mohla cez iterovat viackrat
        self.parameters = list(parameters)
        self.learning_rate = learning_rate

        # Nesterov momentum
        # ukladanie velocities do pamate
        self.momentum = momentum
        if self.momentum is not None:
            self.velocities = []
            for p in self.parameters:
                self.velocities.append(torch.zeros(p.shape, device=p.device, dtype=p.dtype))
        else:
            self.velocities = None

    # zmena vah bez ukladania gradientov
    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.parameters):

                g = p.grad
                if g is None:
                    continue

                # SGD bez momentum
                elif self.momentum is None:
                    # aktualizacia vah: theta = theta - lr * gradient
                    p -= self.learning_rate * g

                # Nesterov momentum
                else:
                    v = self.velocities[i]
                    # aktualizacia velocity: beta * v + gradient
                    new_velocity = self.momentum * v + g
                    self.velocities[i] = new_velocity
                    # aktualizacia vah: theta = theta - lr * (beta * v + gradient)
                    p -= self.learning_rate * (self.momentum * new_velocity + g)
                

    # vynulovanie gradientov, aby ich PyTorch nescitaval pri kazdom batchi
    # pri kazdom backward() prechode su spocitane nove gradienty
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.zero_()

# Adam optimizer
class Adam:
    def __init__(self, parameters, learning_rate, beta1_momentum=0.9, beta2_momentum=0.999, epsilon=1e-8):
        self.parameters = list(parameters)
        self.learning_rate = learning_rate

        self.beta1 = beta1_momentum
        self.beta2 = beta2_momentum
        self.epsilon = epsilon
        self.t = 0

        self.s = []
        self.r = []

        for p in self.parameters:
            self.s.append(torch.zeros(p.shape, device=p.device, dtype=p.dtype))
            self.r.append(torch.zeros(p.shape, device=p.device, dtype=p.dtype))

    def step(self):
        self.t += 1

        with torch.no_grad():
            for i, p in enumerate(self.parameters):

                # gradient aktualneho batchu
                g = p.grad
                if g is None:
                    continue

                # s ← β1s + (1 − β1)g
                self.s[i] = self.beta1 * self.s[i] + (1 - self.beta1) * g
                # r ← β2r + (1 − β2)g^2
                self.r[i] = self.beta2 * self.r[i] + (1 - self.beta2) * (g*g)

                # bias correction
                # s_hat ← s/(1 − β1^t)
                s_estimate = self.s[i] / (1 - self.beta1 ** self.t)
                # r_hat ← r/(1 − β2^t)
                r_estimate = self.r[i] / (1 - self.beta2 ** self.t)

                # θ ← θ − α/(r_hat^(1/2)+ε) * s_hat
                p -= self.learning_rate / (torch.sqrt(r_estimate) + self.epsilon) * s_estimate

    # vynulovanie grad. aktualneho batchu
    # self.s, self.r si Adam pamata
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.zero_()


def main(args: argparse.Namespace) -> dict[str, float]:
    print("Entering")
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
    
    # optimizer -> chcem minimalizovat chybu ucenia (optimizer bude upravovat vahy)
    if args.optimizer == "SGD":
        optimizer = SGD(
            model.parameters(),
            learning_rate=args.learning_rate,
            momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = Adam(
            model.parameters(),
            learning_rate=args.learning_rate
        )
    
    # scheduler -> chcem dynamicky menit rychlost ucenia
    scheduler = None

    # - If `args.decay` is set, then also create a LR scheduler (otherwise, pass `None`).
    #   The scheduler should decay the learning rate from the initial `args.learning_rate`
    #   to the final `args.learning_rate_final`. The `scheduler.step()` is called after
    #   each batch, so the number of scheduler iterations is the number of batches in all
    #   training epochs (note that `len(train)` is the number of batches in one epoch).
    
    #   In all cases (linear/exponential/cosine), you should reach `args.learning_rate_final` just after the training.
    #
    #   If a learning rate schedule is used, the `TrainableModule` automatically logs the
    #   learning rate to the console and to TensorBoard. Additionally, you can find out
    #   the next learning rate to be used by printing `model.scheduler.get_last_lr()[0]`.
    #   Therefore, after the training, this value should be `args.learning_rate_final`.

    if args.decay is not None:
        # pouzi optimizer z Pytorchu nech je kompatibilny so schedulerom
        if args.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.learning_rate,
                momentum=args.momentum if args.momentum is not None else 0.0,
                nesterov=args.momentum is not None
            )

        elif args.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate
            )

        # total_iters = number of batches in one epoch * number of training epochs
        total_iters = len(train) * args.epochs

        #   - for `linear`, use `torch.optim.lr_scheduler.LinearLR` and set `start_factor`,
        #     `end_factor`, and `total_iters` appropriately;
        if args.decay == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                # start = 1.0 * args.learning_rate
                start_factor=1.0,
                end_factor=args.learning_rate_final / args.learning_rate,
                total_iters=total_iters
            )

        #   - for `exponential`, use `torch.optim.lr_scheduler.ExponentialLR` and set `gamma`
        #     appropriately (be careful to compute it using float64 to avoid precision issues);
        elif args.decay == "exponential":
            # α_t = α_initial * gamma^t -> gamma = (α_t / α)^(1/t)
            gamma = np.float64(args.learning_rate_final / args.learning_rate) ** (1 / total_iters)

            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=float(gamma)
            )

        #   - for `cosine`, use `torch.optim.lr_scheduler.CosineAnnealingLR` and set `T_max`
        #     and `eta_min` appropriately.
        elif args.decay == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                # T_max is the maximum number of epochs in a cycle
                T_max=total_iters,
                # eta_min is minimum lr (kde chcem znizit lr)
                eta_min=args.learning_rate_final
            )
        

    model.configure(
        optimizer=optimizer,
        scheduler=scheduler,
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
