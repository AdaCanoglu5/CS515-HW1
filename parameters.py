import argparse


def _parse_hidden_sizes(hidden_sizes_str):
    return [int(x.strip()) for x in hidden_sizes_str.split(",") if x.strip()]


def get_params():
    parser = argparse.ArgumentParser(description="MLP on MNIST")
    parser.add_argument("--mode", choices=["train", "test", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--hidden_sizes", type=str, default="512,256,128")
    parser.add_argument("--activation", choices=["relu", "gelu", "tanh"], default="relu")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_bn", type=int, choices=[0, 1], default=1)
    parser.add_argument("--bn_position", choices=["pre", "post"], default="pre")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler", choices=["none", "step", "cosine"], default="step")
    parser.add_argument("--step_size", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--tmax", type=int, default=None)

    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--l1_lambda", type=float, default=0.0)

    parser.add_argument("--save_path", type=str, default="best_model.pth")
    parser.add_argument("--log_csv", type=str, default="runs/run.csv")
    parser.add_argument("--run_name", type=str, default="")
    args = parser.parse_args()
    hidden_sizes = _parse_hidden_sizes(args.hidden_sizes)

    return {
        # Data
        "data_dir": "./data",
        "num_workers": 2,

        # Model
        "input_size": 784,  # 28x28
        "hidden_sizes": hidden_sizes,
        "num_classes": 10,
        "activation": args.activation,
        "dropout": args.dropout,
        "use_bn": args.use_bn,
        "bn_position": args.bn_position,

        # Training
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "scheduler": args.scheduler,
        "step_size": args.step_size,
        "gamma": args.gamma,
        "tmax": args.tmax if args.tmax is not None else args.epochs,
        "weight_decay": args.weight_decay,
        "l1_lambda": args.l1_lambda,

        # Misc
        "device": args.device,
        "save_path": args.save_path,
        "log_csv": args.log_csv,
        "run_name": args.run_name,
        "log_interval": 100,        # print every N batches

        # CLI
        "mode": args.mode,
    }
