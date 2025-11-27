import argparse
import datetime as dt
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from data_interpreter_Keras_aux import dataInterpreter


def _get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class FitRecConfig:
    # High level experiment configuration
    target_attr: str
    input_attrs: List[str]
    # Keras-style CLI args
    patience: int
    max_epochs: int
    attributes: List[str]
    attr_dim: int
    hidden_dim: int
    lr: float
    batch_size: int
    user_reg: float
    sport_reg: float
    gender_reg: float
    output_reg: float
    include_temporal: bool
    pretrain: bool
    pretrain_file: str
    # Whether to apply context dropout (True for speed, False for heart rate
    context_dropout: bool


class FitRecDataset(Dataset):
    """
    Thin Dataset wrapper around the original dataInterpreter.

    It reproduces the semantics of generator_for_autotrain / dataIteratorSupervised:
    - Same train/valid/test splits
    - Same per-sample construction of inputs/targets
    """

    def __init__(self, endo_reader: dataInterpreter, split: str):
        super().__init__()
        self.endo_reader = endo_reader
        if split == "train":
            self.indices = self.endo_reader.trainingSet
        elif split == "valid":
            self.indices = self.endo_reader.validationSet
        elif split == "test":
            self.indices = self.endo_reader.testSet
        else:
            raise ValueError(f"Invalid split: {split}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        global_idx = self.indices[idx]
        inputs_dict, outputs, _workoutid = self.endo_reader.generateByIdx(global_idx)
        return inputs_dict, outputs.astype(np.float32)


def fitrec_collate(batch: List[Tuple[Dict[str, np.ndarray], np.ndarray]]):
    """
    Collate function that stacks per-sample numpy arrays into batched tensors.
    Keys mirror the original generator_for_autotrain outputs.
    """
    batch_inputs: Dict[str, List[torch.Tensor]] = {}
    targets: List[torch.Tensor] = []

    for inputs_dict, out in batch:
        targets.append(torch.from_numpy(out))  # (T, out_dim)
        for key, value in inputs_dict.items():
            t = torch.from_numpy(value)
            batch_inputs.setdefault(key, []).append(t)

    batched_inputs: Dict[str, torch.Tensor] = {}
    for key, tensors in batch_inputs.items():
        # (B, T, D)
        batched_inputs[key] = torch.stack(tensors, dim=0).float()

    targets_tensor = torch.stack(targets, dim=0).float()  # (B, T, out_dim)
    return batched_inputs, targets_tensor


class FitRecModel(nn.Module):
    """
    PyTorch re-implementation of the Keras LSTM models used in:
    - heart_rate_aux.py
    - speed_aux.py

    The architecture matches the original:
    - Optional user/sport/gender embeddings
    - Optional temporal context via two LSTMs + projection
    - Two stacked LSTMs with dropout
    - SELU activation on the final Dense layer
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_steps: int,
        num_users: int,
        num_sports: int,
        num_genders: int,
        hidden_dim: int,
        user_dim: int,
        sport_dim: int,
        gender_dim: int,
        include_user: bool,
        include_sport: bool,
        include_gender: bool,
        include_temporal: bool,
        context_dropout: bool,
    ):
        super().__init__()
        self.include_user = include_user
        self.include_sport = include_sport
        self.include_gender = include_gender
        self.include_temporal = include_temporal
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim

        layer1_input_dim = input_dim

        # Embeddings (matching Keras: random_normal(stddev=0.01))
        if include_user:
            self.user_embedding = nn.Embedding(num_users, user_dim)
            nn.init.normal_(self.user_embedding.weight, mean=0.0, std=0.01)
            layer1_input_dim += user_dim
        else:
            self.user_embedding = None

        if include_sport:
            self.sport_embedding = nn.Embedding(num_sports, sport_dim)
            nn.init.normal_(self.sport_embedding.weight, mean=0.0, std=0.01)
            layer1_input_dim += sport_dim
        else:
            self.sport_embedding = None

        if include_gender:
            self.gender_embedding = nn.Embedding(num_genders, gender_dim)
            nn.init.normal_(self.gender_embedding.weight, mean=0.0, std=0.01)
            layer1_input_dim += gender_dim
        else:
            self.gender_embedding = None

        # Temporal context (two LSTMs + projection) as in Keras
        self.context_layer_1 = None
        self.context_layer_2 = None
        self.context_projection = None
        self.context_dropout_1 = None
        self.context_dropout_2 = None

        if include_temporal:
            context_dim = hidden_dim
            self.context_layer_1 = nn.LSTM(
                input_size=input_dim + 1, hidden_size=hidden_dim, batch_first=True
            )
            self.context_layer_2 = nn.LSTM(
                input_size=output_dim, hidden_size=hidden_dim, batch_first=True
            )
            if context_dropout:
                self.context_dropout_1 = nn.Dropout(0.1)
                self.context_dropout_2 = nn.Dropout(0.1)
            else:
                self.context_dropout_1 = nn.Identity()
                self.context_dropout_2 = nn.Identity()
            self.context_projection = nn.Linear(2 * hidden_dim, context_dim)
            layer1_input_dim += context_dim

        # Main stacked LSTMs
        self.layer1 = nn.LSTM(
            input_size=layer1_input_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.dropout2 = nn.Dropout(0.2)

        # Output dense + SELU activation
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.SELU()

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        inputs:
            - 'input': (B, T, input_dim)
            - 'user_input': (B, T, 1)  [optional]
            - 'sport_input': (B, T, 1) [optional]
            - 'gender_input': (B, T, 1) [optional]
            - 'context_input_1': (B, T, input_dim + 1) [optional]
            - 'context_input_2': (B, T, output_dim) [optional]
        """
        x = inputs["input"]  # (B, T, input_dim)

        # Attribute embeddings
        if self.include_user and "user_input" in inputs:
            u = inputs["user_input"].long().squeeze(-1)  # (B, T)
            u_emb = self.user_embedding(u)  # (B, T, user_dim)
            x = torch.cat([x, u_emb], dim=-1)

        if self.include_sport and "sport_input" in inputs:
            s = inputs["sport_input"].long().squeeze(-1)
            s_emb = self.sport_embedding(s)
            x = torch.cat([x, s_emb], dim=-1)

        if self.include_gender and "gender_input" in inputs:
            g = inputs["gender_input"].long().squeeze(-1)
            g_emb = self.gender_embedding(g)
            x = torch.cat([x, g_emb], dim=-1)

        # Temporal context
        if self.include_temporal and self.context_layer_1 is not None:
            c1 = inputs["context_input_1"]  # (B, T, input_dim + 1)
            c2 = inputs["context_input_2"]  # (B, T, output_dim)
            ctx1, _ = self.context_layer_1(c1)
            ctx2, _ = self.context_layer_2(c2)
            ctx1 = self.context_dropout_1(ctx1)
            ctx2 = self.context_dropout_2(ctx2)
            ctx = torch.cat([ctx1, ctx2], dim=-1)
            ctx = self.context_projection(ctx)
            x = torch.cat([ctx, x], dim=-1)

        # Stacked LSTMs
        out, _ = self.layer1(x)
        out = self.dropout1(out)
        out, _ = self.layer2(out)
        out = self.dropout2(out)

        out = self.output_layer(out)
        out = self.activation(out)
        return out


class FitRecTorchTrainer:
    """
    Trainer that closely follows the Keras training logic:
    - Same data splits and batching strategy (floor(len(dataset)/batch_size))
    - Same loss (MSE) and metrics (MAE, RMSE)
    - Early stopping with patience on validation loss
    """

    def __init__(self, cfg: FitRecConfig):
        if cfg.pretrain:
            # The original Keras code supports pretraining; this port focuses
            # on the main supervised experiments.
            raise NotImplementedError("Pretraining is not implemented in the PyTorch port.")

        self.cfg = cfg
        self.device = _get_device(force_cpu=False)

        # Paths mirror the original scripts (sans the old 'path' bug)
        self.model_save_root = "./fitrec/model_states/"
        self.logs_root = "./fitrec/logs/"
        self.data_path = "endomondoHR_proper.json"
        self.train_valid_test_fn = self.data_path.split(".")[0] + "_temporal_dataset.pkl"

        self.z_multiple = 5
        self.train_valid_test_split = [0.8, 0.1, 0.1]
        self.trimmed_workout_len = 450
        self.num_steps = self.trimmed_workout_len

        self.include_user = "userId" in cfg.attributes
        self.include_sport = "sport" in cfg.attributes
        self.include_gender = "gender" in cfg.attributes

        # Data
        self.endo_reader = dataInterpreter(
            cfg.input_attrs,
            [cfg.target_attr],
            self.include_user,
            self.include_sport,
            self.include_gender,
            cfg.include_temporal,
            fn=self.data_path,
            scaleVals=True,
            trimmed_workout_len=self.trimmed_workout_len,
            scaleTargets=False,
            trainValidTestSplit=self.train_valid_test_split,
            zMultiple=self.z_multiple,
            trainValidTestFN=self.train_valid_test_fn,
        )
        self.endo_reader.preprocess_data()

        self.input_dim = self.endo_reader.input_dim
        self.output_dim = self.endo_reader.output_dim

        self.train_size = len(self.endo_reader.trainingSet)
        self.valid_size = len(self.endo_reader.validationSet)
        self.test_size = len(self.endo_reader.testSet)

        self._build_model_and_optim()
        self._build_data_loaders()
        self._init_run_metadata()

    def _build_model_and_optim(self):
        num_users = len(self.endo_reader.oneHotMap["userId"])
        num_sports = len(self.endo_reader.oneHotMap["sport"])
        num_genders = len(self.endo_reader.oneHotMap.get("gender", {}))

        self.model = FitRecModel(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            num_steps=self.num_steps,
            num_users=num_users,
            num_sports=num_sports,
            num_genders=num_genders,
            hidden_dim=self.cfg.hidden_dim,
            user_dim=self.cfg.attr_dim,
            sport_dim=self.cfg.attr_dim,
            gender_dim=self.cfg.attr_dim,
            include_user=self.include_user,
            include_sport=self.include_sport,
            include_gender=self.include_gender,
            include_temporal=self.cfg.include_temporal,
            context_dropout=self.cfg.context_dropout,
        ).to(self.device)

        self.user_reg = self.cfg.user_reg
        self.sport_reg = self.cfg.sport_reg
        self.gender_reg = self.cfg.gender_reg
        self.output_reg = self.cfg.output_reg

        self.criterion = nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.cfg.lr)

    def _build_data_loaders(self):
        train_dataset = FitRecDataset(self.endo_reader, split="train")
        valid_dataset = FitRecDataset(self.endo_reader, split="valid")
        test_dataset = FitRecDataset(self.endo_reader, split="test")

        # Match Keras epoch_size = int(len(split) / batch_size)
        # by setting drop_last=True (floor division).
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=fitrec_collate,
        )
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=fitrec_collate,
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            drop_last=True,
            collate_fn=fitrec_collate,
        )

    def _init_run_metadata(self):
        # Build model_file_name the same way as the Keras code
        self.model_file_name_parts: List[str] = list(self.cfg.attributes)
        if self.cfg.include_temporal:
            self.model_file_name_parts.append("context")

        model_run_id = dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.model_file_name_parts.append(model_run_id)
        self.model_file_name = "_".join(self.model_file_name_parts)

        self.model_dir = os.path.join(self.model_save_root, self.model_file_name)
        self.logs_dir = os.path.join(self.logs_root, self.model_file_name)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        self.best_valid_loss = float("inf")
        self.best_epoch = 0
        self.best_model_path = os.path.join(self.model_dir, f"{self.model_file_name}_best.pt")

    def _l2_regularization(self) -> torch.Tensor:
        reg = torch.zeros((), device=self.device)
        if self.include_user and self.user_reg > 0 and self.model.user_embedding is not None:
            reg = reg + self.user_reg * torch.sum(self.model.user_embedding.weight ** 2)
        if self.include_sport and self.sport_reg > 0 and self.model.sport_embedding is not None:
            reg = reg + self.sport_reg * torch.sum(self.model.sport_embedding.weight ** 2)
        if self.include_gender and self.gender_reg > 0 and self.model.gender_embedding is not None:
            reg = reg + self.gender_reg * torch.sum(self.model.gender_embedding.weight ** 2)
        if self.output_reg > 0:
            reg = reg + self.output_reg * torch.sum(self.model.output_layer.weight ** 2)
        return reg

    def _run_epoch(self, loader: DataLoader, training: bool) -> Tuple[float, float, float, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        n_batches = 0

        for batch_inputs, targets in loader:
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
            targets = targets.to(self.device)

            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                preds = self.model(batch_inputs)
                mse = self.criterion(preds, targets)
                reg = self._l2_regularization()
                loss = mse + reg

                mae = torch.mean(torch.abs(preds - targets))
                rmse = torch.sqrt(mse)

                if training:
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item()
            total_mse += mse.item()
            total_mae += mae.item()
            total_rmse += rmse.item()
            n_batches += 1

        if n_batches == 0:
            return 0.0, 0.0, 0.0, 0.0

        return (
            total_loss / n_batches,
            total_mse / n_batches,
            total_mae / n_batches,
            total_rmse / n_batches,
        )

    def train_and_evaluate(self):
        print("Initializing training ...")
        print(
            f"Train/valid/test sizes: {self.train_size}/{self.valid_size}/{self.test_size}, "
            f"batch_size={self.cfg.batch_size}, steps_per_epoch_train={len(self.train_loader)}"
        )

        for epoch in range(1, self.cfg.max_epochs + 1):
            print()
            print("-" * 50)
            print(f"Epoch {epoch}")
            start_time = time.time()

            train_loss, train_mse, train_mae, train_rmse = self._run_epoch(
                self.train_loader, training=True
            )
            print(
                f"| train loss {train_loss:.5f} | train mse {train_mse:.5f} "
                f"| mae {train_mae:.5f} | rmse {train_rmse:.5f}"
            )

            val_loss, val_mse, val_mae, val_rmse = self._run_epoch(
                self.valid_loader, training=False
            )
            elapsed = time.time() - start_time
            print("-" * 80)
            print(
                f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                f"valid loss {val_loss:.5f} | mse {val_mse:.5f} | mae {val_mae:.5f} | rmse {val_rmse:.5f}"
            )
            print("-" * 80)

            improved = val_loss < self.best_valid_loss
            if improved:
                self.best_valid_loss = val_loss
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  Saved new best model to {self.best_model_path}")
            elif epoch - self.best_epoch >= self.cfg.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {self.best_epoch})")
                break

        # Load best model and evaluate on test set
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
            print(f"Loaded best model from {self.best_model_path}")

        test_loss, test_mse, test_mae, test_rmse = self._run_epoch(
            self.test_loader, training=False
        )
        print("-" * 80)
        print(
            f"| test loss {test_loss:.5f} | mse {test_mse:.5f} "
            f"| mae {test_mae:.5f} | rmse {test_rmse:.5f}"
        )
        print("-" * 80)


def build_config_from_args(args: argparse.Namespace, target_attr: str, input_attrs: List[str],
                           context_dropout: bool) -> FitRecConfig:
    return FitRecConfig(
        target_attr=target_attr,
        input_attrs=input_attrs,
        patience=args.patience,
        max_epochs=args.epoch,
        attributes=args.attributes.split(","),
        attr_dim=args.attr_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        user_reg=args.user_reg,
        sport_reg=args.sport_reg,
        gender_reg=args.gender_reg,
        output_reg=args.out_reg,
        include_temporal=args.temporal,
        pretrain=args.pretrain,
        pretrain_file=args.pretrain_file,
        context_dropout=context_dropout,
    )



