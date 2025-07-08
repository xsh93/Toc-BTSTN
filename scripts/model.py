# -*- coding: utf-8 -*-
import os
from datetime import datetime
import copy
import joblib
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn


class ModuleUnit(nn.Module):
    def __init__(self, input_size, output_size, bias=True, hidden_size=None, hidden_layers=None, activation=None, dropout=None):
        super(ModuleUnit, self).__init__()
        fc_ac = {"relu": "nn.ReLU",
                 "leaky_relu": "nn.LeakyReLU",
                 "tanh": "nn.Tanh",
                 "sigmoid": "nn.Sigmoid"
                 }

        model_list = []
        if hidden_layers is None or hidden_layers == 0:
            model_list.append(nn.Linear(input_size, output_size, bias=bias))
        else:
            for n in range(hidden_layers):
                if n == 0:
                    model_list.append(nn.Linear(input_size, input_size, bias=bias))
                    model_list.append(nn.Linear(input_size, hidden_size, bias=bias))
                else:
                    model_list.append(nn.Linear(hidden_size, hidden_size, bias=bias))
                m = model_list[-1]

                if activation is not None and activation["fc_name"] != "":
                    if activation["fc_name"] in fc_ac.keys():
                        if activation["fc_name"] in ["sigmoid", "tanh"]:
                            nn.init.xavier_normal_(m.weight.data, gain=1)
                            if m.bias is not None:
                                m.bias.data.zero_()
                        else:
                            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=activation["fc_name"])

                        if "params" in activation.keys() and activation["params"] != "":
                            _add_module = eval(fc_ac[activation["fc_name"]] + "(" + activation["params"] + ")")
                            model_list.append(_add_module)
                        else:
                            _add_module = eval(fc_ac[activation["fc_name"]] + "()")
                            model_list.append(_add_module)
                    else:
                        raise ValueError("Activation function is not available")

                if dropout is not None and dropout != 0:
                    if 0 <= dropout < 1:
                        model_list.append(nn.Dropout(p=dropout))
                    else:
                        raise ValueError("Error: Incorrect dropout rate!")

            model_list.append(nn.Linear(hidden_size, output_size, bias=bias))
        self.model = nn.Sequential(*model_list)

    def forward(self, inputs):
        output = self.model(inputs)
        return output


class BasicModel:
    def __init__(self):
        self.model = {}
        self.scaler = {}

    # -------------------------------------------------------------------------------------
    @staticmethod
    def _set_device(gpu_id):
        return torch.device("cpu") if gpu_id == -1 \
            else torch.device("cuda:{}".format(gpu_id)) if torch.cuda.is_available() \
            else torch.device("mps:{}".format(gpu_id))

    # -------------------------------------------------------------------------------------
    def _set_model(self, update: list, learning_rate, decay_gamma):
        optimizer, scheduler = {}, {}
        for idx in update:
            optimizer[idx] = torch.optim.Adam(self.model[idx].parameters(), lr=learning_rate)
            scheduler[idx] = torch.optim.lr_scheduler.ExponentialLR(optimizer[idx], gamma=decay_gamma)
        return optimizer, scheduler

    # -------------------------------------------------------------------------------------
    @staticmethod
    def _model_to_device(model: dict, device: torch.device):
        for mod in model.values():
            mod.to(device)
    # -------------------------------------------------------------------------------------
    # @staticmethod
    # def _adjust_learning_rate(optimizer: dict, epoch: int, start_lr: float, decay_epoch: int):
    #     lr = start_lr * (0.1 ** (epoch // decay_epoch))
    #     for opt in optimizer.values():
    #         for param_group in opt.param_groups:
    #             param_group["lr"] = lr

    # -------------------------------------------------------------------------------------
    @staticmethod
    def _create_ground_truth(matrix, output):
        ground = copy.deepcopy(matrix)
        nan_pos = torch.where(torch.isnan(ground))
        tmp_out = output.detach().to("cpu")
        for i in range(len(nan_pos[0])):
            ground[nan_pos[0][i], nan_pos[1][i]] = tmp_out[nan_pos[0][i], nan_pos[1][i]]
        return ground

    # -------------------------------------------------------------------------------------
    def freeze_model(self) -> None:
        for mod in self.model.values():
            for param in mod.parameters():
                param.requires_grad = False

    # -------------------------------------------------------------------------------------
    def save_model(self, opath:str, model_name: Optional[str]=None) -> None:
        model_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') if model_name is None else model_name
        opath = os.path.join(opath, model_name)
        os.makedirs(opath, exist_ok=True)
        for k, v in self.model.items():
            torch.save(v, os.path.join(opath, f"{k}__{model_name}.pth"))
        print("Model saved: {}".format(opath))

    # -------------------------------------------------------------------------------------
    def load_model(self, ipath:str, model_name:str) -> None:
        ipath = os.path.join(ipath, model_name)
        assert os.path.exists(ipath), f"Model path {ipath} does not exist."
        try:
            for file in os.listdir(ipath):
                if file.endswith(".pth"):
                    if file.split("__")[1][:-4] == model_name:
                        model_path = os.path.join(ipath, file)
                        self.model["{}".format(file.split("__")[0])] = torch.load(model_path, weights_only=False)
            self.scaler = joblib.load(os.path.join(ipath, 'scaler.pkl'))
        except Exception as e:
            raise e
        print("Model loaded successfully")

    # -------------------------------------------------------------------------------------
    @staticmethod
    def save_ckpt(opath:str, epoch:int, model:dict, optimizer:dict, scheduler:dict, loss_epoch:pd.DataFrame, model_name=None) -> None:
        model_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') if model_name is None else model_name
        opath = os.path.join(opath, model_name, f"{model_name}__ckpt", "{}__ckpt_epoch_{}".format(model_name, epoch + 1))
        os.makedirs(opath, exist_ok=True)
        for k, v in model.items():
            checkpoint = {
                "epoch": epoch + 1,
                "loss_epoch_dict": loss_epoch.to_dict(),
                "model_state_dict": v.state_dict(),
                "optimizer_state_dict": optimizer[k].state_dict(),
                "scheduler_state_dict": scheduler[k].state_dict(),
            }
            torch.save(checkpoint, os.path.join(opath, "{}__{}__ckpt_epoch_{}.pth".format(k, model_name, epoch + 1)))
        print("Checkpoint saved at epoch {}: {}".format(epoch + 1, opath))

    # -------------------------------------------------------------------------------------
    def load_ckpt(self, ipath:str, epoch:int, model:dict, optimizer:dict, scheduler:dict, loss_epoch:pd.DataFrame, model_name:str) -> None:
        ipath = os.path.join(ipath, model_name, f"{model_name}__ckpt", "{}__ckpt_epoch_{}".format(model_name, epoch))
        assert os.path.exists(ipath), f"Model path {ipath} does not exist."
        try:
            for file in os.listdir(ipath):
                if file.endswith(".pth"):
                    if file.split("__")[1] == model_name and file.split("__")[2][:-4] == f"ckpt_epoch_{epoch}":
                        checkpoint = torch.load(os.path.join(ipath, file))
                        model["{}".format(file.split("__")[0])].load_state_dict(checkpoint["model_state_dict"])
                        optimizer["{}".format(file.split("__")[0])].load_state_dict(checkpoint["optimizer_state_dict"])
                        scheduler["{}".format(file.split("__")[0])].load_state_dict(checkpoint["scheduler_state_dict"])
                        loss_epoch["Epoch"] = pd.Series(checkpoint["loss_epoch_dict"]["Epoch"])
                        loss_epoch["Loss"] = pd.Series(checkpoint["loss_epoch_dict"]["Loss"])

        except Exception as e:
            raise e
        print("Checkpoint loaded successfully")

    # -------------------------------------------------------------------------------------
    @staticmethod
    def countdown(count):
        split1 = torch.where(count <= 0, 1, 0)
        split2 = torch.where(count <= 0, 0, 1)
        count -= 1
        return split1, split2, count

    # # -------------------------------------------------------------------------------------
    # @staticmethod
    # def shift_unit_mx(unit_mx, direction="forward"):
    #     shifted_matrix = torch.zeros_like(unit_mx)
    #     if direction == "forward":
    #         shifted_matrix[:, 1:] = unit_mx[:, :-1]
    #     else:
    #         shifted_matrix[:, :-1] = unit_mx[:, 1:]
    #     return shifted_matrix


class ClippedAdam(torch.optim.Adam):
    def step(self, closure=None):
        super(ClippedAdam, self).step(closure)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.data.clamp_(0, 1)