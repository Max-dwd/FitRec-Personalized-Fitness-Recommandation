import os
import math
import pickle
import datetime
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from data_interpreter_Keras_aux import dataInterpreter


class EndoDataset(Dataset):
    """
    Thin PyTorch Dataset wrapper around the existing dataInterpreter.
    Returns (inputs_dict, targets) where tensors are shaped:
      - input:          (T, input_dim)          float32
      - user_input:     (T, 1)                  int64 (indices)
      - sport_input:    (T, 1)                  int64
      - gender_input:   (T, 1)                  int64
      - context_input_1:(T, input_dim + 1)      float32
      - context_input_2:(T, output_dim)         float32
      - targets:        (T, output_dim)         float32
    DataLoader will batch these to (B, T, ...).
    """

    def __init__(self, endo_reader: dataInterpreter, split: str):
        self.endo_reader = endo_reader
        if split == "train":
            self.indices = endo_reader.trainingSet
        elif split == "valid":
            self.indices = endo_reader.validationSet
        elif split == "test":
            self.indices = endo_reader.testSet
        else:
            raise ValueError(f"Unknown split: {split}")

        self.embed_keys = {"user_input", "sport_input", "gender_input"}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        original_idx = self.indices[idx]
        inputs_dict, outputs, _ = self.endo_reader.generateByIdx(original_idx)

        tensor_inputs: Dict[str, torch.Tensor] = {}
        for key, value in inputs_dict.items():
            arr = np.asarray(value)
            if key in self.embed_keys:
                tensor_inputs[key] = torch.from_numpy(arr).long()
            else:
                tensor_inputs[key] = torch.from_numpy(arr).float()

        targets = torch.from_numpy(np.asarray(outputs)).float()
        return tensor_inputs, targets


class EndoLSTM(nn.Module):
    """
    PyTorch implementation of the original Keras LSTM architecture used in FitRec.
    Supports optional user/sport/gender embeddings and temporal context inputs.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_users: int,
        num_sports: int,
        num_genders: int,
        user_dim: int,
        sport_dim: int,
        gender_dim: int,
        include_user: bool,
        include_sport: bool,
        include_gender: bool,
        include_temporal: bool,
    ) -> None:
        super().__init__()

        self.include_user = include_user
        self.include_sport = include_sport
        self.include_gender = include_gender
        self.include_temporal = include_temporal

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        layer1_input_dim = input_dim

        # Attribute embeddings
        if include_user:
            self.user_embedding = nn.Embedding(num_users, user_dim)
            layer1_input_dim += user_dim
        else:
            self.user_embedding = None

        if include_sport:
            self.sport_embedding = nn.Embedding(num_sports, sport_dim)
            layer1_input_dim += sport_dim
        else:
            self.sport_embedding = None

        if include_gender:
            self.gender_embedding = nn.Embedding(num_genders, gender_dim)
            layer1_input_dim += gender_dim
        else:
            self.gender_embedding = None

        # Temporal context (previous workouts)
        if include_temporal:
            self.context_dim = hidden_dim
            # context_input_1: (T, input_dim + 1), context_input_2: (T, output_dim)
            self.context_lstm_1 = nn.LSTM(
                input_size=input_dim + 1,
                hidden_size=hidden_dim,
                batch_first=True,
            )
            self.context_lstm_2 = nn.LSTM(
                input_size=output_dim,
                hidden_size=hidden_dim,
                batch_first=True,
            )
            self.context_dropout_1 = nn.Dropout(0.1)
            self.context_dropout_2 = nn.Dropout(0.1)
            self.context_projection = nn.Linear(2 * hidden_dim, self.context_dim)

            layer1_input_dim += self.context_dim
        else:
            self.context_dim = 0
            self.context_lstm_1 = None
            self.context_lstm_2 = None
            self.context_dropout_1 = None
            self.context_dropout_2 = None
            self.context_projection = None

        # Main LSTM stack
        self.lstm1 = nn.LSTM(
            input_size=layer1_input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(0.2)

        self.lstm2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(0.2)

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.SELU()

    def forward(
        self,
        main_input: torch.Tensor,
        user_input: torch.Tensor = None,
        sport_input: torch.Tensor = None,
        gender_input: torch.Tensor = None,
        context_input_1: torch.Tensor = None,
        context_input_2: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        main_input:      (B, T, input_dim)
        *_input:         (B, T, 1) for embeddings, (B, T, D) for context tensors
        Returns:
            predictions: (B, T, output_dim)
        """
        x = main_input

        if self.include_user and user_input is not None and self.user_embedding is not None:
            # user_input: (B, T, 1) indices
            user_ids = user_input.squeeze(-1).long()
            user_emb = self.user_embedding(user_ids)
            x = torch.cat([x, user_emb], dim=-1)

        if self.include_sport and sport_input is not None and self.sport_embedding is not None:
            sport_ids = sport_input.squeeze(-1).long()
            sport_emb = self.sport_embedding(sport_ids)
            x = torch.cat([x, sport_emb], dim=-1)

        if self.include_gender and gender_input is not None and self.gender_embedding is not None:
            gender_ids = gender_input.squeeze(-1).long()
            gender_emb = self.gender_embedding(gender_ids)
            x = torch.cat([x, gender_emb], dim=-1)

        if self.include_temporal and context_input_1 is not None and context_input_2 is not None:
            # context_input_1: (B, T, input_dim + 1)
            # context_input_2: (B, T, output_dim)
            ctx1, _ = self.context_lstm_1(context_input_1)
            ctx2, _ = self.context_lstm_2(context_input_2)
            ctx1 = self.context_dropout_1(ctx1)
            ctx2 = self.context_dropout_2(ctx2)
            ctx = torch.cat([ctx1, ctx2], dim=-1)  # (B, T, 2 * hidden_dim)
            ctx = self.context_projection(ctx)      # (B, T, context_dim)
            x = torch.cat([ctx, x], dim=-1)

        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.fc_out(out)
        out = self.activation(out)
        return out


def _run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> Tuple[float, float, float]:
    """
    Run a single epoch over data_loader.
    Returns (loss, mae, rmse) averaged over all elements.
    """
    criterion = nn.MSELoss()
    if optimizer is None:
        model.eval()
        grad_ctx = torch.no_grad()
    else:
        model.train()
        grad_ctx = torch.enable_grad()

    total_loss = 0.0
    total_l1 = 0.0
    total_se = 0.0
    n_samples = 0
    n_elems = 0

    with grad_ctx:
        for batch_inputs, targets in data_loader:
            batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
            targets = targets.to(device)

            batch_size = targets.size(0)
            n_samples += batch_size

            if optimizer is not None:
                optimizer.zero_grad()

            preds = model(
                main_input=batch_inputs["input"],
                user_input=batch_inputs.get("user_input"),
                sport_input=batch_inputs.get("sport_input"),
                gender_input=batch_inputs.get("gender_input"),
                context_input_1=batch_inputs.get("context_input_1"),
                context_input_2=batch_inputs.get("context_input_2"),
            )

            loss = criterion(preds, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * batch_size

            diff = preds - targets
            se = diff.pow(2).sum().item()
            l1 = diff.abs().sum().item()
            elems = diff.numel()

            total_se += se
            total_l1 += l1
            n_elems += elems

    avg_loss = total_loss / max(1, n_samples)
    mae = total_l1 / max(1, n_elems)
    rmse = math.sqrt(total_se / max(1, n_elems)) if n_elems > 0 else 0.0
    return avg_loss, mae, rmse


def run_experiment(
    target_attr: str,
    input_atts: List[str],
    args,
) -> None:
    """
    Shared training entrypoint used by speed and heart-rate scripts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_root = os.path.join(base_dir, "fitrec", "model_states")
    log_root = os.path.join(base_dir, "fitrec", "logs")
    os.makedirs(model_root, exist_ok=True)
    os.makedirs(log_root, exist_ok=True)

    data_path = "endomondoHR_proper.json"
    train_valid_test_fn = data_path.split(".")[0] + "_temporal_dataset.pkl"

    attr_features = [a for a in args.attributes.split(",") if a]
    include_user = "userId" in attr_features
    include_sport = "sport" in attr_features
    include_gender = "gender" in attr_features
    include_temporal = args.temporal

    print(
        f"include user/sport/gender/temporal = "
        f"{include_user}/{include_sport}/{include_gender}/{include_temporal}"
    )

    trimmed_workout_len = 450
    z_multiple = 5
    scale_vals = True
    scale_targets = False
    train_valid_test_split = [0.8, 0.1, 0.1]

    # Build data interpreter and preprocess data
    # Default to a capped number of workouts to avoid OOM on constrained machines.
    # Users can override via --max_workouts (<=0 means use all).
    max_workouts = getattr(args, "max_workouts", 10000)
    if max_workouts is not None and max_workouts <= 0:
        max_workouts = None

    endo_reader = dataInterpreter(
        input_atts,
        [target_attr],
        includeUser=include_user,
        includeSport=include_sport,
        includeGender=include_gender,
        includeTemporal=include_temporal,
        fn=data_path,
        scaleVals=scale_vals,
        trimmed_workout_len=trimmed_workout_len,
        scaleTargets=scale_targets,
        trainValidTestSplit=train_valid_test_split,
        zMultiple=z_multiple,
        trainValidTestFN=train_valid_test_fn,
        max_workouts=max_workouts,
    )
    endo_reader.preprocess_data()

    input_dim = endo_reader.input_dim
    output_dim = endo_reader.output_dim
    train_size = len(endo_reader.trainingSet)
    valid_size = len(endo_reader.validationSet)
    test_size = len(endo_reader.testSet)

    print(f"Input dim: {input_dim}, output dim: {output_dim}")
    print(f"Train/valid/test sizes: {train_size}/{valid_size}/{test_size}")

    # Build datasets and dataloaders
    train_dataset = EndoDataset(endo_reader, "train")
    valid_dataset = EndoDataset(endo_reader, "valid")
    test_dataset = EndoDataset(endo_reader, "test")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    num_users = len(endo_reader.oneHotMap["userId"])
    num_sports = len(endo_reader.oneHotMap["sport"])
    # gender encoding is based on the same machinery; we just use its mapping size
    num_genders = len(endo_reader.oneHotMap["gender"])

    model = EndoLSTM(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        num_users=num_users,
        num_sports=num_sports,
        num_genders=num_genders,
        user_dim=args.attr_dim,
        sport_dim=args.attr_dim,
        gender_dim=args.attr_dim,
        include_user=include_user,
        include_sport=include_sport,
        include_gender=include_gender,
        include_temporal=include_temporal,
    ).to(device)

    # Optimizer (use out_reg as global weight decay)
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.out_reg,
    )

    # Optional pretraining: load state dict if available
    if getattr(args, "pretrain", False) and getattr(args, "pretrain_file", ""):
        pre_name = args.pretrain_file
        pre_dir = os.path.join(model_root, pre_name)
        pre_path = os.path.join(pre_dir, pre_name + "_best.pt")
        if os.path.exists(pre_path):
            state = torch.load(pre_path, map_location=device)
            try:
                model.load_state_dict(state, strict=False)
                print(f"Loaded pretrained weights from {pre_path}")
            except RuntimeError as e:
                print(f"Failed to load pretrained weights from {pre_path}: {e}")
        else:
            print(f"Pretrained model not found at {pre_path}, skipping.")

    # Build model/run name and directories
    model_name_parts = list(attr_features)
    if include_temporal:
        model_name_parts.append("context")
    run_identifier = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    model_name_parts.append(run_identifier)
    model_name = "_".join(model_name_parts)

    model_dir = os.path.join(model_root, model_name)
    log_dir = os.path.join(log_root, model_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    best_model_path = os.path.join(model_dir, model_name + "_best.pt")
    history_path = os.path.join(log_dir, "history.pkl")

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    print("Starting training...")
    for epoch in range(1, args.epoch + 1):
        print("\n" + "-" * 50)
        print(f"Epoch {epoch}")

        train_loss, train_mae, train_rmse = _run_epoch(
            model, train_loader, device, optimizer=optimizer
        )
        val_loss, val_mae, val_rmse = _run_epoch(
            model, valid_loader, device, optimizer=None
        )

        print(
            f"Train - loss: {train_loss:.5f}, MAE: {train_mae:.5f}, RMSE: {train_rmse:.5f}"
        )
        print(
            f"Valid - loss: {val_loss:.5f}, MAE: {val_mae:.5f}, RMSE: {val_rmse:.5f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mae": train_mae,
                "train_rmse": train_rmse,
                "val_loss": val_loss,
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }
        )
        try:
            with open(history_path, "wb") as f:
                pickle.dump(history, f)
        except Exception as e:
            print(f"Failed to save history: {e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved at epoch {epoch} with val_loss={val_loss:.5f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {epochs_without_improvement} epochs)"
                )
                break

    print(f"Best epoch: {best_epoch}, best validation loss: {best_val_loss:.5f}")

    # Evaluate on test set using best model
    if os.path.exists(best_model_path):
        state = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state, strict=False)

    test_loss, test_mae, test_rmse = _run_epoch(
        model, test_loader, device, optimizer=None
    )
    print(
        f"Test - loss: {test_loss:.5f}, MAE: {test_mae:.5f}, RMSE: {test_rmse:.5f}"
    )


