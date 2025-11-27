import os
import argparse

from fitrec_torch_base import run_experiment


def parse():
    parser = argparse.ArgumentParser(description="FitRec heart-rate prediction (PyTorch)")
    parser.add_argument(
        "--patience",
        default=10,
        type=int,
        help="patience for early stopping",
    )
    parser.add_argument(
        "--epoch",
        default=50,
        type=int,
        help="maximum number of epochs",
    )
    parser.add_argument(
        "--attributes",
        default="userId,sport,gender",
        help="comma-separated user/context attributes to include",
    )
    parser.add_argument(
        "--input_attributes",
        default="distance,altitude,time_elapsed",
        help="comma-separated input attributes",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        help="use a pretrained PyTorch model if available",
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="use temporal/context inputs",
    )
    parser.add_argument(
        "--batch_size",
        default=256,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--attr_dim",
        default=5,
        type=int,
        help="embedding dimension for attribute features",
    )
    parser.add_argument(
        "--hidden_dim",
        default=64,
        type=int,
        help="LSTM hidden dimension",
    )
    parser.add_argument(
        "--lr",
        default=0.005,
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "--user_reg",
        default=0.0,
        type=float,
        help="user attribute L2 regularization (unused, kept for compatibility)",
    )
    parser.add_argument(
        "--sport_reg",
        default=0.01,
        type=float,
        help="sport attribute L2 regularization (unused, kept for compatibility)",
    )
    parser.add_argument(
        "--gender_reg",
        default=0.05,
        type=float,
        help="gender attribute L2 regularization (unused, kept for compatibility)",
    )
    parser.add_argument(
        "--out_reg",
        default=0.0,
        type=float,
        help="output layer/global weight decay",
    )
    parser.add_argument(
        "--pretrain_file",
        default="",
        help="name of pretrained model directory (without extension)",
    )
    parser.add_argument(
        "--max_workouts",
        default=10000,
        type=int,
        help="maximum total number of workouts across train/valid/test (<=0 means use all)",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="force running on CPU even if CUDA is available",
    )

    args = parser.parse_args()
    return args


def main(args):
    # Parse input attributes into a list
    input_atts = [s for s in args.input_attributes.split(",") if s]
    # heart_rate prediction task
    run_experiment(target_attr="heart_rate", input_atts=input_atts, args=args)


if __name__ == "__main__":
    cli_args = parse()
    # Honor --force_cpu (if set) before torch gets imported anywhere else
    if getattr(cli_args, "force_cpu", False):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    main(cli_args)


