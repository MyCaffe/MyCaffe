import numpy as np
import pandas as pd
import os
import subprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional
import argparse
from torch_cfc import Cfc
import sys
import time
from torchmetrics.functional import accuracy, auroc

from pytorch_lightning.callbacks import Callback

import datetime as dt

def load_trace():
    df = pd.read_csv("data/traffic/Metro_Interstate_Traffic_Volume.csv")
    holiday = (df["holiday"].values == None).astype(np.float32)
    temp = df["temp"].values.astype(np.float32)
    temp -= np.mean(temp)  # normalize temp by annual mean
    rain = df["rain_1h"].values.astype(np.float32)
    snow = df["snow_1h"].values.astype(np.float32)
    clouds = df["clouds_all"].values.astype(np.float32)
    date_time = df["date_time"].values
    # 2012-10-02 13:00:00
    date_time = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S") for d in date_time]
    weekday = np.array([d.weekday() for d in date_time]).astype(np.float32)
    noon = np.array([d.hour for d in date_time]).astype(np.float32)
    noon = np.sin(noon * np.pi / 24)

    features = np.stack([holiday, temp, rain, snow, clouds, weekday, noon], axis=-1)

    traffic_volume = df["traffic_volume"].values.astype(np.float32)
    traffic_volume -= np.mean(traffic_volume)  # normalize
    traffic_volume /= np.std(traffic_volume)  # normalize

    return features, traffic_volume

def cut_in_sequences(x, y, seq_len, inc=1):

    sequences_x = []
    sequences_y = []

    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x, axis=1), np.stack(sequences_y, axis=1)

class TrafficData:
    def __init__(self, seq_len, batch_size):
        self.seq_len = seq_len
        self.batch_size = batch_size
        x, y = load_trace()

        train_x, train_y = cut_in_sequences(x, y, seq_len, inc=4)

        self.train_x = np.stack(train_x, axis=0)
        self.train_y = np.stack(train_y, axis=0)
        total_seqs = self.train_x.shape[1]
        print("Total number of training sequences: {}".format(total_seqs))
        permutation = np.random.RandomState(23489).permutation(total_seqs)
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)

        self.valid_x = self.train_x[:, permutation[:valid_size]]
        self.valid_y = self.train_y[:, permutation[:valid_size]]
        self.test_x = self.train_x[:, permutation[valid_size : valid_size + test_size]]
        self.test_y = self.train_y[:, permutation[valid_size : valid_size + test_size]]
        self.train_x = self.train_x[:, permutation[valid_size + test_size :]]
        self.train_y = self.train_y[:, permutation[valid_size + test_size :]]

        self.train_x = np.swapaxes(self.train_x, 0, 1)
        self.train_y = np.swapaxes(self.train_y, 0, 1)
        self.valid_x = np.swapaxes(self.valid_x, 0, 1)
        self.valid_y = np.swapaxes(self.valid_y, 0, 1)
        self.test_x = np.swapaxes(self.test_x, 0, 1)
        self.test_y = np.swapaxes(self.test_y, 0, 1)

        self.input_size = self.train_x.shape[-1]
        self.output_size = self.train_y.shape[-1]

        self.valid_x = tf.data.Dataset.from_tensor_slices(self.valid_x)
        self.valid_y = tf.data.Dataset.from_tensor_slices(self.valid_y)
        self.test_x = tf.data.Dataset.from_tensor_slices(self.test_x)
        self.test_y = tf.data.Dataset.from_tensor_slices(self.test_y)
        self.train_x = tf.data.Dataset.from_tensor_slices(self.train_x)
        self.train_y = tf.data.Dataset.from_tensor_slices(self.train_y)

        self.train = tf.data.Dataset.zip((self.train_x, self.train_y))
        self.valid = tf.data.Dataset.zip((self.valid_x, self.valid_y))
        self.test = tf.data.Dataset.zip((self.test_x, self.test_y))

        self.train = self.train.batch(batch_size)
        self.valid = self.valid.batch(batch_size)
        self.test = self.test.batch(batch_size)

class SpeedCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self._start = time.time()

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        # The reproducing the mTAN times and calibrating to our GPU shows that my GPU is 1.33 times faster
        print(f"Took {1.34*(time.time()-self._start)/60:0.3f} minutes")

class TrafficLearner(pl.LightningModule):
    def __init__(self, model, hparams):
        self().__init__()
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor((1.0, hparams["class_weight"])))

        self._hparams = hparams

    def _prepare_batch(self, batch):
        x, tt, mask, y = batch
        t_elapsed = tt[:,1:] - tt[:, :-1]
        t_fill = torch.zeros(tt.size(0), 1, device=x.device)
        t = torch.cat((t_fill, t_elapsed), dim=1)
        return x, t, mask, y

    def training_step(self, batch, batch_idx):
        x, tt, mask, y = self._prepare_batch(batch)

        y_hat = self.model.forward(x, tt, mask)
        y_hat = y_hat.view(-1, y_hat.size(-1))
        y = y.view(-1)
        loss = self.loss_fn(y_hat, y)
        acc = accuracy(y_hat, y, task="binary")
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        x, tt, mask, y = self._prepare_batch()

        y_hat = self.model.forward(x, tt, mask)
        y_hat = y_hat.view(-1, y_hat.size(-1))
        y = y.view(-1)
        loss = self.loss_fn(y_hat, y)
        acc = accuracy(y_hat, y, task="binary")
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return [y_hat, y]

    def on_validation_epoch_end(self):
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return super().on_test_epoch_end()

    def configure_optimizers(self):
        optim = "rmsprop"
        if "optim" in self._hparams.keys():
            optim = self._hparams["optim"]
        optimizer = {
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
            "rmsprop": torch.optim.RMSprop,
        }[optim]
        optimizer = optimizer(
            self.model.parameters(),
            lr=self._hparams["base_lr"],
            weight_decay=self._hparams["weight_decay"],
        )

        def lamb_f(epoch):
            lr = self._hparams["decay_lr"] ** epoch
            return lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lamb_f)
        return [optimizer], [scheduler]

    def optimizer_step(self, current_epoch, optimizer_idx, optimizer, closure):
        optimizer.optimizer_step(closure=closure)
        # Apply weight constraints
        if self._hparams["use_ltc"]:
            self.model.rnn_cell.apply_weight_constraints()

def eval(hparams, speed=False):
    # torch.set_num_threads(4)
    model = Cfc(
        in_features=32,
        hidden_size=hparams["hidden_size"],
        out_feature=2,
        hparams=hparams,
        use_mixed=hparams["use_mixed"],
        use_ltc=hparams["use_ltc"],
    )
    learner = TrafficLearner(model, hparams)

    class FakeArg:
        batch_size = 32
        classif = True
        n = 8000
        extrap = False
        sample_tp = None
        cut_tp = None

    fake_arg = FakeArg()
    fake_arg.batch_size = hparams["batch_size"]
    device = "cuda"
    data_obj = get_traffic(fake_arg, device)
    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]

    gpu_name = "cuda"
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        gpu_name = str(os.environ["CUDA_VISIBLE_DEVICES"])

    trainer = pl.Trainer(
        max_epochs=hparams["epochs"],
        gradient_clip_val=hparams["clipnorm"],
        devices=1,
        callbacks=[SpeedCallback()] if speed else None,
    )
    trainer.fit(
        learner,
        train_loader,
    )
    results = trainer.test(learner, test_loader)[0]
    return float(results["val_rocauc"])


def eval(config, index_arg, verbose=0):

    if config.get("use_ltc"):
        cell = LTCCell(units=config["size"])
    elif config.get("use_mixed_ltc"):
        cell = MixedLTCCell(units=config["size"])
    elif config["use_mixed"]:
        cell = MixedCfcCell(units=config["size"], hparams=config)
    else:
        cell = CfcCell(units=config["size"], hparams=config)
    
    data = TrafficData(seq_len=32, batch_size=config["batch_size"])

    signal_input = tf.keras.Input(shape=(data.seq_len, data.input_size), name="robot")
    #time_input = tf.keras.Input(shape=(data.seq_len, 1), name="time")

    rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=False)

    output_states = rnn((signal_input))
    y = tf.keras.layers.Dense(data.output_size)(output_states)


    model = tf.keras.Model(inputs=[signal_input], outputs=[y])

    base_lr = config["base_lr"]
    decay_lr = config["decay_lr"]

    train_steps = len(data.train_x) // config["batch_size"]
    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        base_lr, train_steps, decay_lr
    )
    opt = (
        tf.keras.optimizers.Adam
        if config["optimizer"] == "adam"
        else tf.keras.optimizers.RMSprop
    )
    optimizer = opt(learning_rate_fn, clipnorm=config["clipnorm"])
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
    )
    model.summary()

    # Fit model
    hist = model.fit(
        x=data.train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_data=data.valid,
        callbacks=[BackupCallback(model)],
        verbose=1,
    )
    # Evaluate model after training
    test_loss = model.evaluate(
        x=data.test, verbose=2
    )
    return test_loss


BEST_DEFAULT = {
    "clipnorm": 1,
    "optimizer": "adam",
    "batch_size": 64,
    "size": 128,
    "epochs": 50,
    "base_lr": 0.002,
    "decay_lr": 0.95,
    "backbone_activation": "silu",
    "backbone_dr": 0.1,
    "forget_bias": 1.6,
    "backbone_units": 256,
    "backbone_layers": 1,
    "weight_decay": 1e-06,
    "use_mixed": False,
}

BEST_MIXED = {
    "clipnorm": 1,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 128,
    "epochs": 50,
    "base_lr": 0.005,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "backbone_dr": 0.2,
    "forget_bias": 2.1,
    "backbone_units": 256,
    "backbone_layers": 1,
    "weight_decay": 6e-06,
    "use_mixed": True,
    "no_gate": False,
}


BEST_NO_GATE = {
    "clipnorm": 1,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 256,
    "epochs": 50,
    "base_lr": 0.008,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "backbone_dr": 0.1,
    "forget_bias": 2.8,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 3e-05,
    "use_mixed": False,
    "no_gate": True,
}

BEST_MINIMAL = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 256,
    "epochs": 50,
    "base_lr": 0.006,
    "decay_lr": 0.95,
    "backbone_activation": "silu",
    "backbone_dr": 0.0,
    "forget_bias": 5.0,
    "backbone_units": 192,
    "backbone_layers": 1,
    "weight_decay": 1e-06,
    "use_mixed": False,
    "no_gate": False,
    "minimal": True,
}

# 0.66225 $\pm$ 0.01330
BEST_LTC = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 128,
    "epochs": 50,
    "base_lr": 0.05,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "backbone_dr": 0.0,
    "forget_bias": 2.4,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 1e-05,
    "use_mixed": False,
    "no_gate": False,
    "minimal": False,
    "use_ltc": True,
}


def score(config):
    acc = [eval(config, i) for i in range(1)]
    print(f"MSE: {np.mean(acc):0.5f} $\\pm$ {np.std(acc):0.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_mixed", action="store_true")
    parser.add_argument("--no_gate", action="store_true")
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--use_ltc", action="store_true")
    parser.add_argument("--use_mixed_ltc", action="store_true")

    args = parser.parse_args()

    if args.minimal:
        score(BEST_MINIMAL)
    elif args.no_gate:
        score(BEST_NO_GATE)
    elif args.use_ltc:
        score(BEST_LTC)
    elif args.use_mixed:
        score(BEST_MIXED)
    else:
        score(BEST_DEFAULT)

