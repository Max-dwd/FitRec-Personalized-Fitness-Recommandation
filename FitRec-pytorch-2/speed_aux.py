import argparse

from fitrec_torch_base import FitRecTorchTrainer, build_config_from_args


def parse():
    parser = argparse.ArgumentParser(description="FitRec speed prediction (PyTorch)")
    parser.add_argument("--patience", default=10, type=int, help="patience for early stop")  # [3,5,10,20]
    parser.add_argument("--epoch", default=50, type=int, help="max epoch")  # [50,100]
    parser.add_argument("--attributes", default="userId,sport,gender", help="input attributes")
    parser.add_argument(
        "--input_attributes",
        default="distance,altitude,time_elapsed",
        help="comma-separated input attributes",
    )
    parser.add_argument("--pretrain", action="store_true", help="use pretrain model (not supported in torch port)")
    parser.add_argument("--temporal", action="store_true", help="use temporal input")
    parser.add_argument("--batch_size", default=256, type=int, help="batch size")
    parser.add_argument("--attr_dim", default=5, type=int, help="attribute dimension")
    parser.add_argument("--hidden_dim", default=64, type=int, help="rnn hidden dimension")
    parser.add_argument("--lr", default=0.005, type=float, help="learning rate")
    parser.add_argument("--user_reg", default=0.0, type=float, help="user attribute L2 regularization")
    parser.add_argument("--sport_reg", default=0.01, type=float, help="sport attribute L2 regularization")
    parser.add_argument("--gender_reg", default=0.05, type=float, help="gender attribute L2 regularization")
    parser.add_argument("--out_reg", default=0.0, type=float, help="final output layer L2 regularization")
    parser.add_argument("--pretrain_file", default="", help="pretrain file (Keras-style, not used here)")
    return parser.parse_args()


def main(args: argparse.Namespace):
    # Original speed_aux.py used a fixed list of attributes but exposed the same default
    # via --input_attributes. Here we honor the CLI while keeping the same default.
    input_attrs = [s.strip() for s in args.input_attributes.split(",") if s.strip()]

    cfg = build_config_from_args(
        args=args,
        target_attr="derived_speed",
        input_attrs=input_attrs,
        # Original speed_aux.py applies dropout on context embeddings
        context_dropout=True,
    )

    trainer = FitRecTorchTrainer(cfg)
    trainer.train_and_evaluate()


if __name__ == "__main__":
    main(parse())
