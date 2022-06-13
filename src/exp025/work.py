import os
import gc
import re
import sys
import copy
import time
import glob
import math
import random
import psutil
import shutil
import typing
import numbers
import warnings
import argparse
from pathlib import Path
from datetime import datetime
from collections.abc import MutableMapping, Iterable
from abc import ABC, ABCMeta, abstractmethod

import json
import yaml

from tqdm.auto import tqdm

import joblib

from box import Box

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    StratifiedGroupKFold,
)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors, NeighborhoodComponentsAnalysis
from sklearn.impute import KNNImputer

from scipy.special import expit as sigmoid
from scipy.stats import pearsonr

import nltk
import transformers
import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
import torchtext
import torchmetrics

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.config import Config

import mlflow

os.environ["TOKENIZERS_PARALLELISM"] = "true"
tqdm.pandas()

# ====================================================
# utils
# ====================================================


def flatten_dict(
    params: typing.Dict[typing.Any, typing.Any], delimiter: str = "/"
) -> typing.Dict[str, typing.Any]:
    """
    https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/loggers/base.py

    Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.
    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.
    Returns:
        Flattened dict.
    Examples:
        >>> LightningLoggerBase._flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> LightningLoggerBase._flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> LightningLoggerBase._flatten_dict({5: {'a': 123}})
        {'5/a': 123}
    """

    def _dict_generator(input_dict, prefixes=None):
        prefixes = prefixes[:] if prefixes else []
        if isinstance(input_dict, typing.MutableMapping):
            for key, value in input_dict.items():
                key = str(key)
                if isinstance(value, (typing.MutableMapping, argparse.Namespace)):
                    value = (
                        vars(value) if isinstance(value, argparse.Namespace) else value
                    )
                    yield from _dict_generator(value, prefixes + [key])
                else:
                    yield prefixes + [key, value if value is not None else str(None)]
        else:
            yield prefixes + [input_dict if input_dict is None else str(input_dict)]

    return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}


def memory_used_to_str():
    pid = os.getpid()
    processs = psutil.Process(pid)
    memory_use = processs.memory_info()[0] / 2.0**30
    return "ram memory gb :" + str(np.round(memory_use, 2))


def seed_everything(seed: int = 42, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = deterministic  # type: ignore


def clear_work(directory):
    shutil.rmtree(directory)
    os.makedirs(directory)


# ====================================================
# plots
# ====================================================
def plot_dist(ytrue, ypred, pred_sigmoid, filename):
    if pred_sigmoid:
        ypred = sigmoid(ypred)
    plt.figure()
    plt.hist(ytrue, alpha=0.5, bins=20)
    plt.hist(ypred, alpha=0.5, bins=20)
    plt.legend(["ytrue", "ypred"])
    plt.savefig(filename)
    plt.close()


def plot_scatter(ytrue, ypred, pred_sigmoid, filename):
    if pred_sigmoid:
        ypred = sigmoid(ypred)
    plt.figure()
    sns.scatterplot(x=ypred, y=ytrue)
    plt.savefig(filename)
    plt.close()


# ====================================================
# extensions
# ====================================================
class Evaluator(ppe.training.extension.Extension):
    priority = ppe.training.extension.PRIORITY_WRITER

    def __init__(self, model, device, metrics, loader, prefix):
        self.model = model
        self.device = device
        self.metrics = metrics
        self.prefix = prefix
        self.loader = loader

    def __call__(self, manager):
        self.model.eval()
        for name in self.metrics.keys():
            self.metrics[name].reset()

        with torch.no_grad():
            for batch in self.loader:
                for k in batch.keys():
                    batch[k] = batch[k].to(self.device)

                yhat = self.model(batch).detach().cpu()

                for name, metric in self.metrics.items():
                    metric.update(yhat, batch["score"].detach().cpu())

        for name, metric in self.metrics.items():
            value = metric.compute()
            ppe.reporting.report({self.prefix + name: value})


# ====================================================
# data processing
# ====================================================
def get_cpc_texts(input_dir):
    CPC = ["A", "B", "C", "D", "E", "F", "G", "H", "Y"]
    contexts = [[c] for c in CPC]
    pattern = "[A-Z]\d+"
    for file_name in os.listdir(os.path.join(input_dir, "CPCSchemeXML202105")):
        result = re.findall(pattern, file_name)
        if result:
            contexts.append(result)
    contexts = sorted(set(sum(contexts, [])))
    results = {}
    for cpc in CPC:
        with open(
            os.path.join(
                input_dir,
                f"CPCTitleList202202/cpc-section-{cpc}_20220201.txt",
            )
        ) as f:
            s = f.read()
        pattern = f"{cpc}\t\t.+"
        result = re.findall(pattern, s)
        pattern = "^" + pattern[:-2]
        cpc_result = re.sub(pattern, "", result[0])
        for context in [c for c in contexts if c[0] == cpc]:
            pattern = f"{context}\t\t.+"
            result = re.findall(pattern, s)
            pattern = "^" + pattern[:-2]
            results[context] = cpc_result + ". " + re.sub(pattern, "", result[0])
    return results


def prepare_fold(df: pd.DataFrame, n_fold: int, seed: int):
    dfx = (
        pd.get_dummies(df, columns=["score"]).groupby(["anchor"], as_index=False).sum()
    )
    cols = [c for c in dfx.columns if c.startswith("score_") or c == "anchor"]
    dfx = dfx[cols]

    mskf = MultilabelStratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)
    labels = [c for c in dfx.columns if c != "anchor"]
    dfx_labels = dfx[labels]
    dfx["fold"] = -1

    for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
        dfx.loc[val_, "fold"] = fold

    fold_array = df.merge(dfx[["anchor", "fold"]], on="anchor", how="left")[
        "fold"
    ].to_numpy()
    df["fold"] = fold_array

    print("#" * 30, "folds", "#" * 30)
    print(df["fold"].value_counts())


class Preprocessor(object):
    def __init__(self, input_nltk_dir) -> None:
        if input_nltk_dir not in nltk.data.path:
            nltk.data.path.append(input_nltk_dir)

        self.stop_words = set(nltk.corpus.stopwords.words("english"))

    @staticmethod
    def untokenize(words):
        # from https://github.com/commonsense/metanl/blob/master/metanl/token_utils.py
        """
        Untokenizing a text undoes the tokenizing operation, restoring
        punctuation and spaces to the places that people expect them to be.
        Ideally, `untokenize(tokenize(text))` should be identical to `text`,
        except for line breaks.
        """
        text = " ".join(words)
        step1 = text.replace("`` ", '"').replace(" ''", '"').replace(". . .", "...")
        step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
        step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
        step4 = re.sub(r" ([.,:;?!%]+)$", r"\1", step3)
        step5 = (
            step4.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")
        )
        step6 = step5.replace(" ` ", " '")
        return step6.strip()

    def remove_stopwords(self, text):
        tokens = nltk.word_tokenize(text)
        text = self.untokenize([t for t in tokens if t.lower() not in self.stop_words])
        return text

    def transform_lower(self, text):
        return text.lower()

    def replace(self, text, f, t):
        return text.replace(f, t)

    def __call__(self, text):
        # text = self.remove_stopwords(text)
        text = self.transform_lower(text)
        text = self.replace(text, ";", ",")

        return text


# ====================================================
# loss
# ====================================================
class MSEWithLogitsLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        loss = F.mse_loss(inputs, targets)

        return loss


# ====================================================
# metrics
# ====================================================
class BCEMetric(torchmetrics.Metric):
    def __init__(self, compute_on_step: bool = True) -> None:
        super().__init__(compute_on_step=compute_on_step)
        self.value = 0
        self.l = 0

    def update(self, yhat, y):
        eps = 1e-07
        yhat = torch.clamp(yhat.view(-1), min=eps, max=1 - eps)
        y = y.view(-1)
        self.value += F.binary_cross_entropy(yhat, y, reduction="sum")
        self.l += int(y.size()[0])

    def compute(self):
        return self.value / self.l


class BCEWithLogitsMetric(torchmetrics.Metric):
    def __init__(self, compute_on_step: bool = True) -> None:
        super().__init__(compute_on_step=compute_on_step)
        self.value = 0
        self.l = 0

    def update(self, yhat, y):
        yhat = yhat.view(-1)
        y = y.view(-1)
        self.value += F.binary_cross_entropy_with_logits(yhat, y, reduction="sum")
        self.l += int(y.size()[0])

    def compute(self):
        return self.value / self.l


class MSEWithLogitsMetric(torchmetrics.Metric):
    def __init__(self, compute_on_step: bool = True) -> None:
        super().__init__(compute_on_step=compute_on_step)
        self.value = 0
        self.l = 0

    def update(self, yhat, y):
        yhat = yhat.view(-1)
        y = y.view(-1)

        yhat = torch.sigmoid(yhat)

        self.value += F.mse_loss(yhat, y, reduction="sum")
        self.l += int(y.size()[0])

    def compute(self):
        return self.value / self.l


class PearsonCorrCoefWithLogitsMetric(torchmetrics.PearsonCorrCoef):
    def __init__(self, compute_on_step: bool = True) -> None:
        super().__init__(compute_on_step=compute_on_step)

    def update(self, yhat, y):
        yhat = yhat.view(-1)
        y = y.view(-1)

        yhat = torch.sigmoid(yhat)

        super().update(yhat, y)

    def compute(self):
        return super().compute()


# ====================================================
# optimizer
# ====================================================
class RAdam(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        degenerated_to_sgd=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if "betas" in param and (
                    param["betas"][0] != betas[0] or param["betas"][1] != betas[1]
                ):
                    param["buffer"] = [[None, None, None] for _ in range(10)]
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=[[None, None, None] for _ in range(10)],
        )
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state["step"] += 1
                buffered = group["buffer"][int(state["step"] % 10)]
                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta2_t = beta2 ** state["step"]
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        ) / (1 - beta1 ** state["step"])
                    elif self.degenerated_to_sgd:
                        step_size = 1.0 / (1 - beta1 ** state["step"])
                    else:
                        step_size = -1
                    buffered[2] = step_size

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(-step_size * group["lr"], exp_avg, denom)
                    p.data.copy_(p_data_fp32)
                elif step_size > 0:
                    if group["weight_decay"] != 0:
                        p_data_fp32.add_(
                            -group["weight_decay"] * group["lr"], p_data_fp32
                        )
                    p_data_fp32.add_(-step_size * group["lr"], exp_avg)
                    p.data.copy_(p_data_fp32)

        return loss


class Lamb(torch.optim.Optimizer):
    # Reference code: https://github.com/cybertronai/pytorch-lamb

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0,
        clamp_value: float = 10,
        adam: bool = False,
        debias: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if weight_decay < 0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if clamp_value < 0.0:
            raise ValueError("Invalid clamp value: {}".format(clamp_value))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.clamp_value = clamp_value
        self.adam = adam
        self.debias = debias

        super(Lamb, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    msg = (
                        "Lamb does not support sparse gradients, "
                        "please consider SparseAdam instead"
                    )
                    raise RuntimeError(msg)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Paper v3 does not use debiasing.
                if self.debias:
                    bias_correction = math.sqrt(1 - beta2 ** state["step"])
                    bias_correction /= 1 - beta1 ** state["step"]
                else:
                    bias_correction = 1

                # Apply bias to lr to avoid broadcast.
                step_size = group["lr"] * bias_correction

                weight_norm = torch.norm(p.data).clamp(0, self.clamp_value)

                adam_step = exp_avg / exp_avg_sq.sqrt().add(group["eps"])
                if group["weight_decay"] != 0:
                    adam_step.add_(p.data, alpha=group["weight_decay"])

                adam_norm = torch.norm(adam_step)
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio = 1
                else:
                    trust_ratio = weight_norm / adam_norm
                state["weight_norm"] = weight_norm
                state["adam_norm"] = adam_norm
                state["trust_ratio"] = trust_ratio
                if self.adam:
                    trust_ratio = 1

                p.data.add_(adam_step, alpha=-step_size * trust_ratio)

        return loss


class SAM(torch.optim.Optimizer):
    # https://github.com/davda54/sam/blob/main/sam.py

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad)
                    .norm(p=2)
                    .to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


# ====================================================
# dataset
# ====================================================
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df=None,
        tokenizer=None,
        max_length=None,
        input_cpc_dir=None,
        input_nltk_dir=None,
    ):
        assert max_length is not None
        self.max_length = max_length

        assert input_cpc_dir is not None
        self.cpc_texts = get_cpc_texts(input_cpc_dir)

        assert input_nltk_dir is not None
        self.preprocessor = Preprocessor(input_nltk_dir=input_nltk_dir)

        self.tokenizer = tokenizer

        if df is not None:
            self.anchors = df["anchor"].apply(self.preprocessor).to_numpy()
            self.targets = df["target"].apply(self.preprocessor).to_numpy()
            self.cpc_titles = (
                df["context"]
                .apply(lambda x: self.cpc_texts[x[0]])
                .apply(self.preprocessor)
                .to_numpy()
            )
            self.cpc_contexts = (
                df["context"]
                .apply(lambda x: self.cpc_texts[x])
                .apply(self.preprocessor)
                .to_numpy()
            )
            self.scores = df["score"].to_numpy()
            self.length = len(df)

    def __len__(self):
        return self.length

    def lazy_init(self, df=None):
        """Reset Members"""
        if df is not None:
            self.anchors = df["anchor"].apply(self.preprocessor).to_numpy()
            self.targets = df["target"].apply(self.preprocessor).to_numpy()
            self.cpc_sections = ("[" + df["context"].str[0] + "]").to_numpy()
            self.cpc_titles = (
                df["context"]
                .apply(lambda x: self.cpc_texts[x[0]])
                .apply(self.preprocessor)
                .to_numpy()
            )
            self.cpc_contexts = (
                df["context"]
                .apply(lambda x: self.cpc_texts[x])
                .apply(self.preprocessor)
                .to_numpy()
            )
            self.scores = df["score"].to_numpy()
            self.length = len(df)

    def __getitem__(self, idx):

        # select sep_token
        # sep = self.tokenizer.sep_token
        sep = "[s]"

        anchor = self.anchors[idx]
        target = self.targets[idx]
        cpc_section = self.cpc_sections[idx]
        cpc_title = self.cpc_titles[idx]
        cpc_context = self.cpc_contexts[idx]

        text = f"{cpc_section} {sep} {anchor} {sep} {target} {sep} {cpc_context}"
        output = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            truncation=True,
        )

        score = self.scores[idx].astype("float32")
        output["score"] = score

        return output


class Collate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output = dict()

        output["input_ids"] = [sample["input_ids"] for sample in batch]
        output["attention_mask"] = [sample["attention_mask"] for sample in batch]

        # calculate max token length of this batch
        batch_max = max([len(input_ids) for input_ids in output["input_ids"]])

        # add padding
        if self.tokenizer.padding_side == "right":
            output["input_ids"] = [
                s + (batch_max - len(s)) * [self.tokenizer.pad_token_id]
                for s in output["input_ids"]
            ]
            output["attention_mask"] = [
                s + (batch_max - len(s)) * [0] for s in output["attention_mask"]
            ]
        else:
            output["input_ids"] = [
                (batch_max - len(s)) * [self.tokenizer.pad_token_id] + s
                for s in output["input_ids"]
            ]
            output["attention_mask"] = [
                (batch_max - len(s)) * [0] + s for s in output["attention_mask"]
            ]

        # convert to tensors
        output["input_ids"] = torch.tensor(
            output["input_ids"],
            dtype=torch.long,
        )
        output["attention_mask"] = torch.tensor(
            output["attention_mask"],
            dtype=torch.long,
        )

        output["score"] = torch.tensor(
            [sample["score"] for sample in batch],
            dtype=torch.float32,
        )

        return output


# ====================================================
# tokenizer
# ====================================================
def get_tokenizer(tokenizer_path, tokenizer_params):
    config = transformers.AutoConfig.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )
    config.update(tokenizer_params)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        config=config,
        local_files_only=True,
    )

    # add tokens
    sectoks = ["[" + s + "]" for s in "ABCDEFGHs"]
    tokenizer.add_special_tokens({"additional_special_tokens": sectoks})

    return tokenizer


# ====================================================
# model
# ====================================================
class SimpleHead(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout_rate,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, last_hidden_state):
        x = self.layer_norm(last_hidden_state[:, 0, :])
        x = self.dropout(x)
        output = self.linear(x)
        return output


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(AttentionHead, self).__init__()
        self.W = nn.Linear(in_features, hidden_features)
        self.V = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        attention_scores = self.V(torch.tanh(self.W(x)))
        attention_scores = torch.softmax(attention_scores, dim=1)
        attentive_x = attention_scores * x
        attentive_x = attentive_x.sum(axis=1)
        return attentive_x


class MaskAddedAttentionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MaskAddedAttentionHead, self).__init__()
        self.W = nn.Linear(in_features, hidden_features)
        self.V = nn.Linear(hidden_features, out_features)

    def forward(self, x, attention_mask):
        attention_scores = self.V(torch.tanh(self.W(x)))
        attention_scores = attention_scores + attention_mask
        attention_scores = torch.softmax(attention_scores, dim=1)
        attentive_x = attention_scores * x
        attentive_x = attentive_x.sum(axis=1)
        return attentive_x


class AttentionPoolHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.GELU(),
            nn.Linear(in_features, 1),
        )
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, mask):
        w = self.attention(x).float()
        w[mask == 0] = float("-inf")
        w = torch.softmax(w, 1)
        x = torch.sum(w * x, dim=1)
        output = self.linear(x)
        return output


class MeanPoolingHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        norm_mean_embeddings = self.layer_norm(mean_embeddings)
        logits = self.linear(norm_mean_embeddings).squeeze(-1)

        return logits


class MeanMaxPoolingHead(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        self.linear = nn.Linear(in_features * 2, 1)
        super().__init__()

    def forward(self, last_hidden_state):
        mean_pooling_embeddings = torch.mean(last_hidden_state, 1)
        _, max_pooling_embeddings = torch.max(last_hidden_state, 1)
        mean_max_embeddings = torch.cat(
            (
                mean_pooling_embeddings,
                max_pooling_embeddings,
            ),
            1,
        )
        logits = self.linear(mean_max_embeddings)  # twice the hidden size

        return logits


class LSTMHead(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=in_features,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(in_features * 2)
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, last_hidden_state):
        x, _ = self.lstm(last_hidden_state, None)
        x = self.layer_norm(x[:, -1, :])
        output = self.linear(x)
        return output


class GRUHead(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        dropout,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=in_features,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(in_features * 2)
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, last_hidden_state):
        x, _ = self.gru(last_hidden_state, None)
        x = self.layer_norm(x[:, -1, :])
        output = self.linear(x)
        return output.squeeze(-1)


class CNNHead(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_size,
        kernel_size,
    ):
        super().__init__()
        self.cnn1 = nn.Conv1d(
            in_features, hidden_size, kernel_size=kernel_size, padding=1
        )
        self.cnn2 = nn.Conv1d(
            hidden_size, out_features, kernel_size=kernel_size, padding=1
        )
        self.relu = nn.ReLU()

    def forward(self, last_hidden_state):
        x = last_hidden_state.permute(0, 2, 1)
        x = self.cnn1(x)
        x = self.relu(x)
        x = self.cnn2(x)
        x, _ = torch.max(x, 2)
        return x


class Model(nn.Module):
    def __init__(
        self,
        encoder_name,
        encoder_path,
        encoder_params,
        embedding_length,
        head_type,
        head_params,
    ):
        super().__init__()

        # model
        config = transformers.AutoConfig.from_pretrained(
            encoder_path,
            local_files_only=True,
        )
        config.update(encoder_params)
        self.encoder = transformers.AutoModel.from_pretrained(
            encoder_path,
            config=config,
            local_files_only=True,
        )
        self.encoder.resize_token_embeddings(embedding_length)

        self.head = eval(head_type)(in_features=config.hidden_size, **head_params)
        for module in self.head.modules():
            self._init_weights(module, config)

    def forward(self, x):
        last_hidden_state = self.encoder(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
        )["last_hidden_state"]
        if isinstance(
            self.head,
            (MeanPoolingHead, MaskAddedAttentionHead, AttentionPoolHead),
        ):
            output = self.head(last_hidden_state, x["attention_mask"])
        else:
            output = self.head(last_hidden_state)
        return output.view(-1)

    def _init_weights(self, module, config):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


def get_optimizer_params(model, encoder_lr, head_lr, weight_decay=0.0):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p
                for n, p in model.encoder.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.encoder.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "lr": encoder_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.head.named_parameters()],
            "lr": head_lr,
            "weight_decay": 0.0,
        },
    ]
    return optimizer_parameters


# ====================================================
# train fold
# ====================================================
def train_fold(CFG, fold):
    print("#" * 30, f"fold: {fold}", "#" * 30)

    torch.backends.cudnn.benchmark = CFG["/training/benchmark"]
    seed_everything(CFG["/globals/seed"], deterministic=CFG["/training/deterministic"])
    device = torch.device(CFG["/training/device"])

    accumulation_steps = CFG["/training/accumulate_gradient_batchs"]
    max_grad_norm = CFG["training/max_grad_norm"]

    train_dataloader = CFG["/dataloader/train"]
    valid_dataloader = CFG["/dataloader/valid"]

    model = CFG["/model"]
    model.to(device)
    optimizer = CFG["/optimizer"]
    scheduler = CFG["/scheduler"]
    loss_func = CFG["/loss"]

    use_amp = CFG["/training/use_amp"]
    scaler = amp.GradScaler(enabled=use_amp)

    manager = CFG["/manager"]
    for ext in CFG["/extensions"]:
        manager.extend(**ext)

    while not manager.stop_trigger:
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            with manager.run_iteration():
                for k in batch.keys():
                    batch[k] = batch[k].to(device)

                with amp.autocast(use_amp):
                    yhat = model(batch)

                    loss = loss_func(yhat, batch["score"])
                    if accumulation_steps > 1:
                        loss = loss / accumulation_steps

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_grad_norm,
                )

                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                ppe.reporting.report({"loss": loss.item()})

    del train_dataloader, valid_dataloader, model, device
    gc.collect()
    torch.cuda.empty_cache()


# ====================================================
# inference
# ====================================================
def inference(stage, model_path, CFG):
    dataloader = CFG[f"/dataloader/{stage}"]

    model = CFG["/model"]
    device = CFG["/training/device"]
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    pred_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="predicting..."):
            for k in batch.keys():
                batch[k] = batch[k].to(device)

            yhat = model(batch)
            pred_list.append(yhat.detach().cpu().numpy())

    del model, device
    torch.cuda.empty_cache()
    gc.collect()

    return np.concatenate(pred_list)


# ====================================================
# main
# ====================================================
def training_main(PRE_EVAL_CFG, results):
    train_df = pd.read_csv(Path(PRE_EVAL_CFG["globals"]["input_dir"], "train.csv"))

    # debug?
    if PRE_EVAL_CFG["globals"]["debug"]:
        train_df = train_df.sample(
            n=1000,
            random_state=PRE_EVAL_CFG["globals"]["seed"],
        ).reset_index(drop=True)

    print(f"train length = {len(train_df)}")

    # folds
    prepare_fold(
        df=train_df,
        n_fold=PRE_EVAL_CFG["globals"]["n_fold"],
        seed=PRE_EVAL_CFG["globals"]["seed"],
    )

    # prepare
    oof_df = pd.DataFrame({"id": train_df["id"], "score": np.zeros(len(train_df))})
    for fold in range(PRE_EVAL_CFG["globals"]["n_fold"]):
        train_idx = train_df.loc[train_df["fold"] != fold].index
        valid_idx = train_df.loc[train_df["fold"] == fold].index

        FOLD_PRE_EVAL_CFG = copy.deepcopy(PRE_EVAL_CFG)
        FOLD_PRE_EVAL_CFG["globals"]["fold"] = fold
        CFG = Config(FOLD_PRE_EVAL_CFG, types=CONFIG_TYPES)
        model_path = CFG["/model_filename"]

        CFG["/dataset/train"].lazy_init(df=train_df.loc[train_idx])
        CFG["/dataset/valid"].lazy_init(df=train_df.loc[valid_idx])

        train_fold(CFG=CFG, fold=fold)

        oof_prediction = inference(stage="valid", model_path=model_path, CFG=CFG)
        oof_df.loc[valid_idx, "score"] = oof_prediction

        torch.cuda.empty_cache()
        gc.collect()

        results.model_paths.append(model_path)

    oof_df.to_csv("oof.csv", index=False)

    metric = CFG["/metric/metric"]
    metric.reset()
    metric(torch.tensor(oof_df["score"]), torch.tensor(train_df["score"]))
    validation_score = float(metric.compute().detach().cpu().numpy())

    print(f"validation score: {validation_score}")
    results.metrics.update({"valid_score": validation_score})

    # plots
    pred_sigmoid = isinstance(metric, PearsonCorrCoefWithLogitsMetric)
    plot_dist(
        train_df["score"].to_numpy(),
        oof_df["score"].to_numpy(),
        pred_sigmoid=pred_sigmoid,
        filename="oof_dist_plot.png",
    )
    plot_scatter(
        train_df["score"].to_numpy(),
        oof_df["score"].to_numpy(),
        pred_sigmoid=pred_sigmoid,
        filename="oof_scatter_plot.png",
    )


def inference_main(PRE_EVAL_CFG, results):
    test_df = pd.read_csv(Path(PRE_EVAL_CFG["globals"]["input_dir"], "test.csv"))
    test_df["score"] = 0

    submission_df = pd.DataFrame({"id": test_df["id"], "score": np.zeros(len(test_df))})

    for fold, model_path in enumerate(results.model_paths):
        print("#" * 30, f"fold: {fold}", "#" * 30)
        print("#" * 30, f"model path: {model_path}", "#" * 30)

        CFG = Config(PRE_EVAL_CFG, types=CONFIG_TYPES)

        CFG["/dataset/test"].lazy_init(df=test_df)

        test_prediction = inference(stage="test", model_path=model_path, CFG=CFG)
        submission_df["score"] += test_prediction

    submission_df["score"] /= len(results.model_paths)
    submission_df.to_csv("submission.csv", index=False)
    results.metrics.update({"public_lb": np.nan})
    results.metrics.update({"private_lb": np.nan})


# ====================================================
# save results
# ====================================================


def save_results_main(PRE_EVAL_CFG, results):
    print(results)

    client = mlflow.tracking.MlflowClient(PRE_EVAL_CFG["mlflow"]["save_dir"])
    try:
        experiment_id = client.create_experiment(
            PRE_EVAL_CFG["mlflow"]["experiment_name"]
        )
    except:
        experiment = client.get_experiment_by_name(
            PRE_EVAL_CFG["mlflow"]["experiment_name"]
        )
        experiment_id = experiment.experiment_id

    run = client.create_run(experiment_id)

    # desc
    if os.path.exists(str(globals().get("__file__"))):
        dir_ = os.path.join(os.path.dirname(__file__), PRE_EVAL_CFG["RUN_NAME"])
        os.makedirs(dir_, exist_ok=True)
        shutil.copy(__file__, os.path.join(dir_, "work.py"))
        with open(os.path.join(dir_, "desc.txt"), "w") as f:
            f.write(PRE_EVAL_CFG["RUN_DESC"])

    client.log_param(run.info.run_id, "name", PRE_EVAL_CFG["RUN_NAME"])
    client.log_param(run.info.run_id, "desc", PRE_EVAL_CFG["RUN_DESC"])

    # params
    results.params = PRE_EVAL_CFG
    for key, value in flatten_dict(results.params).items():
        client.log_param(run.info.run_id, key, value)

    # metric
    for key, value in results.metrics.items():
        client.log_metric(run.info.run_id, key, value)

    # artifacts
    filename = "results.pkl"
    joblib.dump(results, filename)
    for filename in glob.glob("./*"):
        client.log_artifact(run.info.run_id, filename)

    return results


def premain(directory):
    clear_work(directory)
    os.chdir(directory)
    if os.path.exists(str(globals().get("__file__"))):
        shutil.copy(__file__, "work.py")


# ====================================================
# config
# ====================================================
CONFIG_STRING = """

RUN_NAME: exp025
RUN_DESC: "deberta-v3-base"

globals:
  fold: null # indicate when training
  seed: 42
  n_fold: 5
  work_dir: /workspaces/us-patent-phrase-to-phrase-matching/work
  input_dir: ../input/us-patent-phrase-to-phrase-matching
  input_cpc_dir: ../input/cpc-data
  input_nltk_dir: ../input/nltk-downloads
  input_huggingface_dir: ../input/huggingface-models
  debug: False

training:
  device: cuda
  use_amp: True
  benchmark: False
  deterministic: True
  max_epochs: 10
  max_grad_norm: 2.0
  accumulate_gradient_batchs: 1
  steps_per_epoch: {type: integer_div_ceil, x: {type: __len__, obj: "@/dataloader/train"}, y: "@/training/accumulate_gradient_batchs"}

mlflow:
  save_dir: ../mlruns
  experiment_name: experiments
  save_results: True



### model, tokenizer

model_filename: {type: str_concat, ls: [model_, "@/model/encoder_name/", _fold, "@/globals/fold", .pth]}


model:
  type: Model
  encoder_name: 
    type: get_encoder_name
    path: "@/model/encoder_path"
  encoder_path:
    type: path_join
    ls: ["@/globals/input_huggingface_dir", "microsoft/deberta-v3-base"]
  encoder_params:
    hidden_dropout_prob: 0.1
    attention_probs_dropout_prob: 0.1
  embedding_length: {type: __len__, obj: "@/tokenizer"}
  head_type: SimpleHead
  head_params:
    out_features: 1
    dropout_rate: 0.10 # SimpleHead
    # hidden_features: 1024 # AttentionHead, MaskAddedAttentionHead
    # hidden_size: 256 # CNNHead
    # kernel_size: 8 # CNNHead
    # dropout: 0.0 # LSTMHead, GRUHead

tokenizer:
  type: get_tokenizer
  tokenizer_path: "@/model/encoder_path"
  tokenizer_params: {}


### dataset, dataloader

dataset:
  max_length: 256
  train:
    type: Dataset
    df: null  # set by lazy_init
    tokenizer: "@/tokenizer"
    max_length: "@/dataset/max_length"
    input_cpc_dir: "@/globals/input_cpc_dir"
    input_nltk_dir: "@/globals/input_nltk_dir"
  valid:
    type: Dataset
    df: null  # set by lazy_init
    tokenizer: "@/tokenizer"
    max_length: "@/dataset/max_length"
    input_cpc_dir: "@/globals/input_cpc_dir"
    input_nltk_dir: "@/globals/input_nltk_dir"
  test:
    type: Dataset
    df: null  # set by lazy_init
    tokenizer: "@/tokenizer"
    max_length: "@/dataset/max_length"
    input_cpc_dir: "@/globals/input_cpc_dir"
    input_nltk_dir: "@/globals/input_nltk_dir"

dataloader:
  train: 
    type: DataLoader
    dataset: "@/dataset/train"
    batch_size: 16
    num_workers: 2
    shuffle: True
    pin_memory: True
    drop_last: True
    collate_fn:
      type: Collate
      tokenizer: "@/tokenizer"
  valid: 
    type: DataLoader
    dataset: "@/dataset/valid"
    batch_size: 16
    num_workers: 2
    shuffle: False
    pin_memory: False
    drop_last: False
    collate_fn:
      type: Collate
      tokenizer: "@/tokenizer"
  test: 
    type: DataLoader
    dataset: "@/dataset/test"
    batch_size: 16
    num_workers: 1
    shuffle: False
    pin_memory: False
    drop_last: False
    collate_fn:
      type: Collate
      tokenizer: "@/tokenizer"


### optimizer, scheduler

optimizer:
  type: AdamW # [SGD, Adam, AdamW, RAdam, Lamb, SAM]
  params:
    type: get_optimizer_params
    model: "@/model"
    encoder_lr: 1.0e-05
    head_lr:    1.0e-05
    weight_decay: 0.00
  # betas: [0.9, 0.999] # Lamb, RAdam
  # eps: 1.0e-07 # Lamb, RAdam
  # adam: True # Lamb
  # debias: False # Lamb
  # rho: 0.05 # SAM
  # adaptive: False # SAM
  # base_optimizer: {type: getattr, obj: {type: eval, name: torch.optim}, name: Adam} # SAM

scheduler:
  type: OneCycleLR # [None, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR]
  optimizer: "@/optimizer"
  # mode: min # ReduceLROnPlateau
  # factor: 0.1 # ReduceLROnPlateau
  # patience: 2 # ReduceLROnPlateau
  # eps: 1.0e-08 # ReduceLROnPlateau
  # T_max: {type: __len__, obj: "@/dataloader/train"} # CosineAnnealingLR
  # T_0: {type: __len__, obj: "@/dataloader/train"} # CosineAnnealingWarmRestarts
  # T_mult: 2 # CosineAnnealingWarmRestarts
  # eta_min: 1.0e-12 # CosineAnnealingLR, CosineAnnealingWarmRestarts
  
  max_lr: [5.0e-05, 5.0e-05, 5.0e-05] # lr of [encoder, encoder(nodecay), head] # OneCycleLR
  pct_start: 0.1 # OneCycleLR
  steps_per_epoch: "@/training/steps_per_epoch" # OneCycleLR
  epochs: "@/training/max_epochs" # OneCycleLR
  anneal_strategy: cos # OneCycleLR
  div_factor: 1.0e+02 # OneCycleLR
  final_div_factor: 1 # OneCycleLR
  
  verbose: False


### loss, metrics

## with sigmoid

loss: {type: MSEWithLogitsLoss} # {MSEWithLogitsLoss, BCEWithLogitsLoss}

metric: 
  mse_loss: {type: MSEWithLogitsMetric, compute_on_step: False}
  bce_loss: {type: BCEWithLogitsMetric, compute_on_step: False}
  metric: {type: PearsonCorrCoefWithLogitsMetric, compute_on_step: False}


## without sigmoid

# loss: {type: MSELoss} # {MSELoss}

# metric: 
#   mse_loss: {type: MSEMetric, compute_on_step: False}
#   bce_loss: {type: BCEMetric, compute_on_step: False}
#   metric: {type: PearsonCorrCoefMetric, compute_on_step: False}



### manager, extensions

manager:
  type: ExtensionsManager
  models: "@/model"
  optimizers: "@/optimizer"
  max_epochs: "@/training/max_epochs"
  iters_per_epoch: {type: __len__, obj: "@/dataloader/train"}
  out_dir: "@/globals/work_dir"

extensions:
  # snapshot
  - extension: {type: snapshot, target: "@/model", n_retains: 1, filename: "@/model_filename"}
    trigger: {type: MaxValueTrigger, key: "valid/metric", trigger: [1, epoch]}
  # log
  - extension: {type: observe_lr, optimizer: "@/optimizer"}
    trigger: {type: IntervalTrigger, period: 1, unit: iteration}
  - extension: {type: LogReport, filename: {type: str_concat, ls: [log_fold, "@/globals/fold"]}}
    trigger: {type: IntervalTrigger, period: 1, unit: epoch}
  - extension: {type: PlotReport, y_keys: lr, x_key: iteration, filename: {type: str_concat, ls: [lr_fold, "@/globals/fold", .png]}}
    trigger: {type: IntervalTrigger, period: 1, unit: epoch}
  - extension: {type: PlotReport, y_keys: [valid/mse_loss], x_key: epoch, filename: {type: str_concat, ls: [mse_loss_fold, "@/globals/fold", .png]}}
    trigger: {type: IntervalTrigger, period: 1, unit: epoch}
  - extension: {type: PlotReport, y_keys: [valid/bce_loss], x_key: epoch, filename: {type: str_concat, ls: [bce_loss_fold, "@/globals/fold", .png]}}
    trigger: {type: IntervalTrigger, period: 1, unit: epoch}
  - extension: {type: PlotReport, y_keys: [valid/metric], x_key: epoch, filename: {type: str_concat, ls: [metrics_fold, "@/globals/fold", .png]}}
    trigger: {type: IntervalTrigger, period: 1, unit: epoch}
  - extension: {type: PrintReport, entries: [epoch, iteration, lr, loss, valid/mse_loss, valid/bce_loss, valid/metric, elapsed_time]}
    trigger: {type: IntervalTrigger, period: 1, unit: epoch}
  - extension: {type: ProgressBar, update_interval: 1}
    trigger: {type: IntervalTrigger, period: 1, unit: iteration}
  # evaluator
  - extension: {type: Evaluator, loader: "@/dataloader/valid", prefix: "valid/", model: "@/model", metrics: "@/metric", device: "@/training/device"}
    trigger: {type: IntervalTrigger, period: 1, unit: epoch}
"""


CONFIG_TYPES = {
    # # utils
    "__len__": lambda obj: len(obj),
    "getattr": lambda obj, name: getattr(obj, name),
    "eval": lambda name: eval(name),
    "get_encoder_name": lambda path: path.split("/")[-1],
    "str_concat": lambda ls: "".join([str(x) for x in ls]),
    "method_call": lambda obj, method: getattr(obj, method)(),
    "path_join": lambda ls: os.path.join(*ls),
    "integer_div": lambda x, y: x // y,
    "integer_div_ceil": lambda x, y: (x + y - 1) // y,
    # # DataFrame
    "DataFrame": pd.DataFrame,
    # # Dataset, DataLoader
    "Preprocessor": Preprocessor,
    "get_tokenizer": get_tokenizer,
    "Dataset": Dataset,
    "DataLoader": torch.utils.data.DataLoader,
    "Collate": Collate,
    # # Model
    "Model": Model,
    "SimpleHead": SimpleHead,
    "AttentionHead": AttentionHead,
    "MaskAddedAttentionHead": MaskAddedAttentionHead,
    "MeanPoolingHead": MeanPoolingHead,
    "MeanMaxPoolingHead": MeanMaxPoolingHead,
    "LSTMHead": LSTMHead,
    "GRUHead": GRUHead,
    "CNNHead": CNNHead,
    "get_optimizer_params": get_optimizer_params,
    # # Optimizer
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "RAdam": RAdam,
    "Lamb": Lamb,
    "SAM": SAM,
    # # Scheduler
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR,
    # # Loss
    "BCELoss": torch.nn.BCELoss,
    "MSELoss": torch.nn.MSELoss,
    "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss,
    "MSEWithLogitsLoss": MSEWithLogitsLoss,
    # # Metric
    "BCEWithLogitsMetric": BCEWithLogitsMetric,
    "MSEWithLogitsMetric": MSEWithLogitsMetric,
    "PearsonCorrCoefWithLogitsMetric": PearsonCorrCoefWithLogitsMetric,
    "BCEMetric": BCEMetric,
    "MSEMetric": torchmetrics.MeanSquaredError,
    "PearsonCorrCoefMetric": torchmetrics.PearsonCorrCoef,
    # # PPE Extensions
    "ExtensionsManager": ppe.training.ExtensionsManager,
    "observe_lr": ppe.training.extensions.observe_lr,
    "LogReport": ppe.training.extensions.LogReport,
    "PlotReport": ppe.training.extensions.PlotReport,
    "PrintReport": ppe.training.extensions.PrintReport,
    "ProgressBar": ppe.training.extensions.ProgressBar,
    "snapshot": ppe.training.extensions.snapshot,
    "LRScheduler": ppe.training.extensions.LRScheduler,
    "MinValueTrigger": ppe.training.triggers.MinValueTrigger,
    "MaxValueTrigger": ppe.training.triggers.MaxValueTrigger,
    "EarlyStoppingTrigger": ppe.training.triggers.EarlyStoppingTrigger,
    "IntervalTrigger": ppe.training.triggers.IntervalTrigger,
    "Evaluator": Evaluator,
}

if __name__ == "__main__":

    PRE_EVAL_CFG = yaml.safe_load(CONFIG_STRING)

    # kaggle環境ならTrue
    if "KAGGLE_URL_BASE" in os.environ.keys():
        PRE_EVAL_CFG["globals"]["work_dir"] = "."
        PRE_EVAL_CFG["mlflow"]["save_dir"] = "./mlruns"

    # debug?
    if PRE_EVAL_CFG["globals"]["debug"]:
        PRE_EVAL_CFG["globals"]["n_fold"] = 3
        PRE_EVAL_CFG["training"]["max_epochs"] = 3
        PRE_EVAL_CFG["mlflow"]["experiment_name"] = "debug"

    results = Box(model_paths=[], metrics={})

    premain("/workspaces/us-patent-phrase-to-phrase-matching/work")

    training_main(PRE_EVAL_CFG, results)
    inference_main(PRE_EVAL_CFG, results)

    if PRE_EVAL_CFG["mlflow"]["save_results"]:
        results = save_results_main(PRE_EVAL_CFG, results)
