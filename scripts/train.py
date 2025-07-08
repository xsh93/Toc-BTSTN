# -*- coding: utf-8 -*-
import os
import copy
import joblib
import json
from typing import Optional
from decimal import Decimal
from tqdm import trange
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from .model import ModuleUnit, BasicModel, ClippedAdam
from .utils import DataReader

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class BTSTN(BasicModel):
    def __init__(
            self,
            o_features: int,
            p_features: int,
            o_hid_dims: int,
            g_inner: Optional[int] = None,
            g_layer: Optional[int] = None,
            g_activation: Optional[dict] = None,
            g_dropout: float = 0,
            d_inner: Optional[int] = None,
            d_layer: Optional[int] = None,
            d_activation: Optional[dict] = None,
            d_dropout: float = 0,
            unit: float = 1,
            dropna: bool = False,
            scale: Optional[str] = None,
    ):

        super().__init__()
        self.__dict__.update(locals())
        self.model = self._init_modules()
        self.scaler = None

    # -------------------------------------------------------------------------------------
    # Initialize models
    # -------------------------------------------------------------------------------------
    def _init_modules(self) -> dict:
        return {
            "D": ModuleUnit(
                input_size=self.o_hid_dims,
                output_size=self.o_features,
                bias=True,
                hidden_size=self.d_inner,
                hidden_layers=self.d_layer,
                activation=self.d_activation,
                dropout=self.d_dropout
            ),

            "Gf": ModuleUnit(
                input_size=self.o_hid_dims + self.p_features,
                output_size=self.o_hid_dims,
                bias=True,
                hidden_size=self.g_inner,
                hidden_layers=self.g_layer,
                activation=self.g_activation,
                dropout=self.g_dropout
            ),

            "Gb": ModuleUnit(
                input_size=self.o_hid_dims + self.p_features,
                output_size=self.o_hid_dims,
                bias=True,
                hidden_size=self.g_inner,
                hidden_layers=self.g_layer,
                activation=self.g_activation,
                dropout=self.g_dropout
            ),
        }

    # -------------------------------------------------------------------------------------
    # Methods: fit()
    # -------------------------------------------------------------------------------------
    def fit(self,
            odata: pd.DataFrame,
            pdata: pd.DataFrame,
            max_gap: int = 1,
            learning_rate: float = 0.001,
            decay_gamma: Optional[float] = None,
            epoch: int = 1000,
            patience: int = 50,
            batch_size: Optional[int] = None,
            gpu_id: int = -1,
            saving_path: Optional[str] = None,
            saving_prefix: Optional[str] = None,
            ckpt_path: Optional[str] = None,
            ckpt_prefix: Optional[str] = None,
            ckpt_freq: Optional[int] = None,
            ckpt_resume_epoch: Optional[int] = None,
            ) -> pd.DataFrame:

        assert 0 < learning_rate <= 1, "Learning rate must be between 0 and 1."
        # assert patience <= epoch, "[patience] should be less than or equal to [epoch]"

        # prepare dataset
        device = self._set_device(gpu_id)
        data = DataReader(odata=odata, pdata=pdata, unit=self.unit, dropna=self.dropna, scaler=self.scale)
        self.scaler = data.scaler
        joblib.dump(self.scaler, os.path.join(saving_path, saving_prefix, 'scaler.pkl'))
        dataset = self._prepare_dataset(data, max_gap, batch_size)

        # prepare model
        self.model["F"] = ModuleUnit(input_size=data.shape[0], output_size=self.o_hid_dims, bias=False)
        optimizer, scheduler = self._set_model(["F", "D", "Gf", "Gb"], learning_rate, decay_gamma)
        loss_epoch = pd.DataFrame(columns=["Epoch", "Loss"])
        if ckpt_resume_epoch:
            assert ckpt_resume_epoch > 0, "ckpt_resume_epoch must be greater than 0."
            self.load_ckpt(ckpt_path, ckpt_resume_epoch, self.model, optimizer, scheduler, loss_epoch, ckpt_prefix)
        criterion = torch.nn.MSELoss()

        # train model
        train_loss = self._train(
            dataset,
            optimizer,
            scheduler,
            loss_epoch,
            criterion,
            device,
            epoch,
            patience,
            "fitting",
            ckpt_path,
            ckpt_prefix,
            ckpt_freq,
            ckpt_resume_epoch,
        )

        # save model
        if saving_path:
            self.save_model(saving_path, saving_prefix)

        print("Training completed !")

        return train_loss

    # -------------------------------------------------------------------------------------
    def _prepare_dataset(self, data, max_gap, batch_size):
        forward_data = self._process_pair_data(data, max_gap, "forward")
        backward_data = self._process_pair_data(data, max_gap, "backward")
        dataset = TensorDataset(*(forward_data + backward_data))
        batch_size = len(dataset) if batch_size is None else batch_size
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        return [dataloader, max_gap]

    # -------------------------------------------------------------------------------------
    @staticmethod
    def _process_pair_data(data: DataReader, max_gap: Optional[int] = None, direction: str = "forward"):
        _, pair_o_mx, pair_p_mx = data.batch_pair(max_gap, direction=direction)
        gap = torch.FloatTensor(pair_o_mx[0])
        o_unit = torch.FloatTensor(pair_o_mx[1])
        o_matrix = torch.FloatTensor(pair_o_mx[2])
        p_matrix = torch.FloatTensor(pair_p_mx)
        return gap, o_unit, o_matrix, p_matrix

    # -------------------------------------------------------------------------------------
    def _train(self, dataset, optimizer, scheduler, loss_epoch, criterion, device, epoch, patience, process, ckpt_path, ckpt_prefix, ckpt_freq, ckpt_resume_epoch):
        dataloader, max_gap = dataset
        # loss_epoch = pd.DataFrame(columns=["Epoch", "Loss"])
        count_patience = patience
        best_loss = float("inf")
        best_model = self.model
        best_optimizer = optimizer
        best_scheduler = scheduler
        epoch_start = ckpt_resume_epoch if ckpt_resume_epoch is not None else 0

        self._model_to_device(self.model, device)
        with trange(epoch_start, epoch, ncols=100) as t:
            for epo in t:
                t.set_description("==> {}".format(process))
                running_loss = self._train_epoch(dataloader, optimizer, max_gap, criterion, device) * 1e6
                loss_epoch.loc[epo] = [epo + 1, running_loss]
                t.set_postfix(current_loss="{:.5f}".format(running_loss), minimum_loss="{:.5f}".format(best_loss))

                if running_loss < best_loss:
                    best_loss = running_loss
                    best_model = copy.deepcopy(self.model)
                    best_optimizer = copy.deepcopy(optimizer)
                    best_scheduler = copy.deepcopy(scheduler)
                    count_patience = patience
                else:
                    count_patience -= 1
                    if count_patience == 0:
                        break

                for slr in scheduler.values():
                    slr.step()

                if ckpt_path and ckpt_prefix and ckpt_freq:
                    if (epo + 1) % ckpt_freq == 0:
                        self.save_ckpt(
                            ckpt_path,
                            epo,
                            best_model,
                            best_optimizer,
                            best_scheduler,
                            loss_epoch,
                            ckpt_prefix,
                        )

        self.model = best_model
        return loss_epoch

    # -------------------------------------------------------------------------------------
    def _train_epoch(self, dataloader, optimizer, max_gap, criterion, device):
        running_loss = 0
        for data in dataloader:
            output, ground = self._train_flow(data[0:4], max_gap, device, "forward")
            output_back, ground_back = self._train_flow(data[4:8], max_gap, device, "backward")
            output_cat = torch.cat([output, output_back], dim=0)
            ground_cat = torch.cat([ground, ground_back], dim=0)

            loss = criterion(output_cat, ground_cat)
            loss.backward()
            for opt in optimizer.values():
                opt.step()
                opt.zero_grad()

            running_loss += loss.item()

        return running_loss / len(dataloader)

    # -------------------------------------------------------------------------------------
    def _train_flow(self, data, max_gap, device, direction):
        gap, o_unit, o_matrix, p_matrix = [t.to(device) for t in data]
        gout = self.model["F"](o_unit)
        dout = torch.zeros_like(gout)
        sp1, sp2, next_gap = self.countdown(gap)
        for i in range(max_gap):
            identity = gout
            dout += torch.multiply(gout, sp1)
            pout = p_matrix[:, i * self.p_features: (i + 1) * self.p_features]
            op_data = torch.cat((gout, pout), dim=1)
            gout = self.model["Gf"](op_data) if direction == "forward" else self.model["Gb"](op_data)
            gout += identity
            gout = torch.multiply(gout, sp2)
            sp1, sp2, next_gap = self.countdown(next_gap)

        dout += torch.multiply(gout, sp1)
        output = self.model["D"](dout)
        ground = self._create_ground_truth(o_matrix, output)
        ground = ground.to(device)

        return output, ground

    #######################################################################################
    # Methods: forecast()
    #######################################################################################
    def forecast(self,
                 odata: pd.DataFrame,
                 pdata: pd.DataFrame,
                 sdata: pd.DataFrame,
                 start: int = -1,
                 reset: bool = True,
                 max_gap: int = 1,
                 learning_rate: float = 0.001,
                 decay_gamma: Optional[float] = None,
                 epoch: int = 1000,
                 patience: int = 50,
                 batch_size: Optional[int] = None,
                 gpu_id: int = -1,
                 ) -> pd.DataFrame:
        
        assert 0 < learning_rate <= 1, "Learning rate must be between 0 and 1."
        # assert patience <= epoch, "[patience] should be less than or equal to [epoch]"

        device = self._set_device(gpu_id)
        data = DataReader(odata=odata, pdata=pdata, unit=self.unit, dropna=self.dropna, scaler=self.scaler)
        sdata_norm = torch.FloatTensor(self.scaler["p_scaler"].transform(sdata.iloc[:, 2:]))

        if reset:
            dataset = self._prepare_dataset(data, max_gap, batch_size)
            self.model["F"] = ModuleUnit(input_size=data.shape[0], output_size=self.o_hid_dims, bias=False)
            optimizer, scheduler = self._set_model(["F"], learning_rate, decay_gamma)
            criterion = torch.nn.MSELoss()
            _loss_epoch = pd.DataFrame(columns=["Epoch", "Loss"])
            _ = self._train(dataset, optimizer, scheduler, _loss_epoch, criterion, device, epoch, patience, "forecasting", None, None, None, None)

        o_unit = torch.FloatTensor(data.unit_matrix[start])
        gout = self.model["F"](o_unit)
        # output = self.model["D"](gout).reshape(1, -1)
        output = torch.empty((0, self.o_features))
        for i in range(sdata_norm.shape[0]):
            identity = gout
            sout = sdata_norm[i, :]
            op_data = torch.cat((gout, sout), dim=0)
            gout = self.model["Gf"](op_data)
            gout += identity
            dout = self.model["D"](gout).reshape(1, -1)
            output = torch.concatenate((output, dout), dim=0)

        output = output.detach().numpy()
        output = self.scaler["o_scaler"].inverse_transform(output)
        forecast_df = pd.concat([sdata.iloc[:, 0],
                                 sdata.iloc[:, 1] + self.unit,
                                 pd.DataFrame(output, columns=data.data["observe"].columns[2:])
                                 ], axis=1)
        return forecast_df

    #######################################################################################
    # Methods: search_scheme()
    #######################################################################################
    def search_scheme(self,
                      odata: pd.DataFrame,
                      pdata: pd.DataFrame,
                      rdata: pd.DataFrame,
                      start: int = -1,
                      pred_step: int = 1,
                      n_scheme: int = 1,
                      custom_scheme: Optional[list] = None,
                      reset: bool = True,
                      max_gap: int = 1,
                      learning_rate: float = 0.001,
                      decay_gamma: Optional[float] = None,
                      epoch: int = 1000,
                      patience: int = 50,
                      batch_size: Optional[int] = None,
                      gpu_id: int = -1,
                      ) -> pd.DataFrame:

        assert 0 < learning_rate <= 1, "Learning rate must be between 0 and 1."
        # assert patience <= epoch, "[patience] should be less than or equal to [epoch]"

        device = self._set_device(gpu_id)
        data = DataReader(odata=odata, pdata=pdata, unit=self.unit, dropna=self.dropna, scaler=self.scaler)
        rdata = torch.FloatTensor(self.scaler["o_scaler"].transform(rdata))

        if reset:
            dataset = self._prepare_dataset(data, max_gap, batch_size)
            self.model["F"] = ModuleUnit(input_size=data.shape[0], output_size=self.o_hid_dims, bias=False)
            optimizer, scheduler = self._set_model(["F"], learning_rate, decay_gamma)
            criterion = torch.nn.MSELoss()
            _loss_epoch = pd.DataFrame(columns=["Epoch", "Loss"])
            _ = self._train(dataset, optimizer, scheduler, _loss_epoch, criterion, device, epoch, patience, "monitoring", None, None, None, None)

        s_unit = torch.FloatTensor(1 * np.eye(pred_step, dtype=float))
        if n_scheme > 0:
            if custom_scheme is not None:
                # assert len(custom_scheme) + 1 == n_scheme, "The custom_scheme and n_schemes are not equal."
                assert max(custom_scheme) <= pred_step, "The maximum custom_scheme must be less than the pred_step."
                _idx = custom_scheme.copy()
                _idx.insert(0, 0)
            else:
                _step = pred_step // n_scheme
                _idx = [i * _step for i in range(n_scheme)]

            _s_unit = np.zeros((pred_step, pred_step))
            for i in range(len(_idx) - 1):
                _s_unit[_idx[i]:_idx[i + 1], _idx[i]] = 1
            _s_unit[_idx[-1]:, _idx[-1]] = 1
            s_unit = torch.FloatTensor(_s_unit)

        # if n_scheme == 1:
        #     _s_unit = np.zeros((pred_step, pred_step))
        #     _s_unit[:, 0] = 1
        #     s_unit = torch.FloatTensor(_s_unit)

        s_module = ModuleUnit(input_size=pred_step, output_size=self.p_features, bias=False)
        # s_optimizer = torch.optim.Adam(s_module.parameters(), lr=learning_rate)
        s_optimizer = ClippedAdam(s_module.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        loss_epoch = pd.DataFrame(columns=["Epoch", "Loss"])
        patience_count = patience
        best_loss = float("inf")
        best_s_module = s_module
        o_unit = torch.FloatTensor(data.unit_matrix[start])
        with trange(epoch, ncols=100) as t:
            for epoch in t:
                t.set_description("==> {}".format("searching"))
                s_optimizer.zero_grad()
                gout = self.model["F"](o_unit)
                for i in range(pred_step):
                    identity = gout
                    # pout = torch.sigmoid(s_module(s_unit[i, :]))
                    pout = s_module(s_unit[i, :])
                    op_data = torch.cat((gout, pout), dim=0)
                    gout = self.model["Gf"](op_data)
                    gout += identity
                dout = self.model["D"](gout).reshape(1, -1)

                # Calculate training_loss
                ground = self._create_ground_truth(rdata, dout)
                loss = criterion(dout, ground)
                loss.backward()

                s_optimizer.step()
                running_loss = loss.item()
                loss_epoch.loc[epoch] = [epoch + 1, running_loss]
                t.set_postfix(current_loss="{:.5f}".format(running_loss), minimum_loss="{:.5f}".format(best_loss))

                # Check for best model
                if running_loss < best_loss:
                    best_loss = running_loss
                    best_s_module = copy.deepcopy(s_module)
                    patience_count = patience
                else:
                    patience_count -= 1
                    if patience_count == 0:
                        break
        # s_output = torch.sigmoid(best_s_module(s_unit)).detach().numpy()
        s_output = best_s_module(s_unit).detach().numpy()
        s_output = self.scaler["p_scaler"].inverse_transform(s_output)
        # ltime = data.data_raw["combine"]["time"].iloc[-1]
        # scheme_df = pd.concat([pd.DataFrame([data.data_raw["combine"]["batch"].unique()] * pred_step),
        #                        pd.DataFrame(np.arange(ltime + self.unit, ltime + self.unit * pred_step, self.unit)),
        #                        pd.DataFrame(s_output)
        #                        ], axis=1)
        scheme_df = pd.concat([pd.DataFrame(["batch x"] * pred_step),
                               pd.DataFrame(np.arange(0, float(Decimal("{}".format(self.unit)) * Decimal(pred_step)), self.unit)),
                               pd.DataFrame(s_output)
                               ], axis=1)
        scheme_df.columns = ["batch", "time"] + list(data.data["perturb"].columns[2:])
        return scheme_df

    #######################################################################################
    # Methods: load_pretrained()
    #######################################################################################
    @classmethod
    def load_pretrained(cls, ipath, model_name):
        try:
            with open(os.path.join(ipath, model_name, model_name + '.json'), 'r', encoding='utf-8') as f:
                params = json.load(f)

            obj = cls(**params["model_params"])
            obj.load_model(ipath, model_name)
        except Exception as e:
            raise e
        return obj