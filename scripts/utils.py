# -*- coding: utf-8 -*-
from typing import List, Union, Optional
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.simplefilter(action="ignore", category=FutureWarning)

# -------------------------------------------------------------------------------------
# DataType definition
# -------------------------------------------------------------------------------------
DataType = Union[pd.DataFrame, List[pd.DataFrame]]
BatchType = Union[str, List[str]]
TimeType = Union[float, List[float]]
ScalerType = Union[str, dict]

"""
DataReader
Description：process observation and perturbation data
Parameters：
"""


class DataReader:
    def __init__(self,
                 odata: DataType,
                 pdata: DataType,
                 unit: float,
                 dropna: bool = False,
                 scaler: Optional[ScalerType] = None
                 ):
        self.init_params = {"unit": unit, "dropna": dropna, "scaler": scaler}
        self.calc_params = self._process_data(odata, pdata)
        self.show_info()

    # -------------------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------------------
    def _process_data(self, odata, pdata) -> dict:
        odata_df = self._merge_data(odata, self.init_params["dropna"])
        pdata_df = self._merge_data(pdata, self.init_params["dropna"])
        combine_df = pd.merge(odata_df, pdata_df, on=["batch", "time"], how="outer")
        combine_df.iloc[:, 2:] = combine_df.iloc[:, 2:].astype(float)
        batch_num = combine_df["batch"].nunique()
        obs_num, ovar_num, pvar_num = combine_df.shape[0], odata_df.shape[1] - 2, pdata_df.shape[1] - 2
        self._format_data(combine_df, ovar_num, pvar_num)
        return {"batch": batch_num, "obs": obs_num, "ovar": ovar_num, "pvar": pvar_num}

    # -------------------------------------------------------------------------------------
    @staticmethod
    def _merge_data(data: DataType, dropna: bool) -> pd.DataFrame:
        data_lst = data if isinstance(data, list) else [data]
        format_data_lst = []
        for d in data_lst:
            d.columns = d.columns.str.replace(d.columns[0], "batch")
            d.columns = d.columns.str.replace(d.columns[1], "time")
            format_data_lst.append(d)
        merge_dataframe = pd.concat(format_data_lst, ignore_index=True, axis=0)
        merge_dataframe.sort_values(by=["batch", "time"], ascending=[True, True], axis=0, inplace=True)
        if dropna:
            merge_dataframe.dropna(thresh=3, inplace=True)
        merge_dataframe.reset_index(drop=True, inplace=True)
        return merge_dataframe

    # -------------------------------------------------------------------------------------
    def _format_data(self, combine_df_raw, ovar_num, pvar_num):
        observe_df_raw = combine_df_raw.iloc[:, :ovar_num + 2]
        perturb_df_raw = combine_df_raw.iloc[:, [0, 1] + list(range(ovar_num + 2, ovar_num + pvar_num + 2))]
        self.data_raw = {"observe": observe_df_raw, "perturb": perturb_df_raw, "combine": combine_df_raw}
        self.unit_matrix, self.matrix, self.scaler = self._create_matrix(observe_df_raw.iloc[:, 2:], perturb_df_raw.iloc[:, 2:])

        time_unit = combine_df_raw["time"] / self.init_params["unit"]
        time_unit = time_unit.round(0).astype(int)
        combine_df = pd.concat([combine_df_raw["batch"],
                                time_unit,
                                pd.DataFrame(self.matrix["combined"], columns=combine_df_raw.columns[2:])
                                ], axis=1)
        observe_df = combine_df[observe_df_raw.columns]
        perturb_df = combine_df[perturb_df_raw.columns]
        self.data = {"observe": observe_df, "perturb": perturb_df, "combine": combine_df}

    # -------------------------------------------------------------------------------------
    def _create_matrix(self, odata, pdata):
        observe_mx, observe_scaler = self._normalize(odata, self.init_params["scaler"], target="o")
        perturb_mx, perturb_scaler = self._normalize(pdata, self.init_params["scaler"], target="p")
        combined_mx = np.concatenate((observe_mx, perturb_mx), axis=1)
        unit_matrix = 1 * np.eye(combined_mx.shape[0], dtype=float)
        matrix = {"observe": observe_mx, "perturb": perturb_mx, "combined": combined_mx}
        return unit_matrix, matrix, {"o_scaler": observe_scaler, "p_scaler": perturb_scaler}

    # -------------------------------------------------------------------------------------
    @staticmethod
    def _normalize(data: pd.DataFrame, method: Optional[ScalerType] = None, target: str = "o"):
        if method is None:
            scaler = None
            matrix = np.array(data)
        elif isinstance(method, str):
            assert method in ("MinMax", "Standard"), "Normalization Method Not Supported!"
            scaler = StandardScaler() if method == "Standard" else MinMaxScaler(
                feature_range=(0, 1)) if method == "MinMax" else None
            matrix = scaler.fit_transform(data)
        else:
            scaler = method["o_scaler"] if target == "o" else method["p_scaler"]
            matrix = scaler.transform(data)
        return matrix, scaler

    # -------------------------------------------------------------------------------------
    def show_info(self):
        print("Note: DataReader[{} observations x {} variates x {} perturbations, {} batch in total]".format(
            self.calc_params["obs"], self.calc_params["ovar"], self.calc_params["pvar"], self.calc_params["batch"])
        )

    # -------------------------------------------------------------------------------------
    # Supported methods
    # -------------------------------------------------------------------------------------
    # 筛选特定批次与时间点的数据，获得新的DataReader对象
    def query(self, batch: BatchType, time: TimeType):
        batch = [batch] if not isinstance(batch, list) else batch
        time = [time] if not isinstance(time, list) else time

        # 筛选满足query条件的行
        select_index = (self.data_raw["combine"]["batch"].isin(batch)) & (self.data_raw["combine"]["time"].isin(time))
        query_observe_data_raw = self.data_raw["observe"].copy()
        query_observe_data_raw = query_observe_data_raw[select_index]
        query_observe_data_raw.reset_index(drop=True, inplace=True)

        query_perturb_data_raw = self.data_raw["perturb"].copy()
        query_perturb_data_raw = query_perturb_data_raw[select_index]
        query_perturb_data_raw.reset_index(drop=True, inplace=True)

        dr = DataReader(odata=query_observe_data_raw,
                        pdata=query_perturb_data_raw,
                        unit=self.init_params["unit"],
                        dropna=self.init_params["dropna"],
                        scaler=self.init_params["scaler"]
                        )
        return dr

    # -------------------------------------------------------------------------------------
    # 根据指定的最大间隔数进行组合
    def batch_pair(self, max_gap: int, batch: Optional[BatchType] = None, direction: str = "forward"):
        batch = self.data["observe"]["batch"].drop_duplicates().tolist() if batch is None else batch
        batch = [batch] if not isinstance(batch, list) else batch
        assert direction in ["forward", "backward"], ValueError("Incorrect direction.")

        select_batches = self.data["combine"]["batch"].isin(batch)
        o_unit_mx = self.unit_matrix[np.where(select_batches)]
        o_matrix = self.matrix["observe"][np.where(select_batches)]
        p_matrix = self.matrix["perturb"][np.where(select_batches)]

        pair_df = pd.DataFrame(columns=["batch", "start", "end", "gap"])
        pair_gap_mx = np.empty(shape=(0, 1), dtype=int)
        pair_o_unit_mx = np.empty(shape=(0, o_unit_mx.shape[1]), dtype=int)
        pair_o_matrix = np.empty(shape=(0, o_matrix.shape[1]), dtype=float)
        pair_p_matrix = np.empty(shape=(0, p_matrix.shape[1] * max_gap), dtype=float)

        count = 0
        for bt in batch:
            bt_df = self.data["combine"][self.data["combine"]["batch"] == bt].reset_index(drop=True)

            # -------------------------------------------------------------------------------------
            # time matrix
            bt_time_series = bt_df["time"].sort_values().to_numpy()
            bt_time_count = len(bt_time_series)
            cur_max_gap = np.max(bt_time_series) - np.min(bt_time_series) if max_gap is None else max_gap

            bt_time_mx1 = self._dup_cols(np.reshape(bt_time_series, (bt_time_count, 1)), 0, bt_time_count - 1).astype(int)
            bt_time_mx2 = self._dup_rows(np.reshape(bt_time_series, (1, bt_time_count)), 0, bt_time_count - 1).astype(int)
            subtract_mx = bt_time_mx2 - bt_time_mx1 if direction == "forward" else bt_time_mx1 - bt_time_mx2
            select_index = (subtract_mx >= 0) & (subtract_mx <= cur_max_gap)

            # -------------------------------------------------------------------------------------
            # pair dataframe
            start_time = bt_time_mx1[select_index]
            end_time = bt_time_mx2[select_index]
            gap = subtract_mx[select_index]
            pair_df_cur_batch = pd.DataFrame({
                "batch": bt,
                "start": start_time,
                "end": end_time,
                "gap": gap
            })
            pair_df = pd.concat([pair_df, pair_df_cur_batch], ignore_index=True)

            # -------------------------------------------------------------------------------------
            # pair_o_unit_mx
            o_unit_index = np.searchsorted(bt_time_series, bt_time_mx1[select_index])
            pair_o_unit_mx_cur_batch = o_unit_mx[count:count + bt_time_count][o_unit_index]
            pair_o_unit_mx = np.row_stack((pair_o_unit_mx, pair_o_unit_mx_cur_batch))

            # -------------------------------------------------------------------------------------
            # pair_o_matrix
            o_matrix_index = np.searchsorted(bt_time_series, bt_time_mx2[select_index])
            pair_o_matrix_cur_batch = o_matrix[count:count + bt_time_count][o_matrix_index]
            pair_o_matrix = np.row_stack((pair_o_matrix, pair_o_matrix_cur_batch))

            # -------------------------------------------------------------------------------------
            # pair_gap_mx
            pair_gap_mx_cur_batch = gap.reshape(len(gap), 1)
            pair_gap_mx = np.row_stack((pair_gap_mx, pair_gap_mx_cur_batch))

            # -------------------------------------------------------------------------------------
            # pair_p_unit_mx
            # pair_p_unit_mx = pair_o_unit_mx.copy()

            # -------------------------------------------------------------------------------------
            # pair_p_matrix
            p_matrix_index = np.searchsorted(bt_time_series, bt_time_mx1[select_index])
            select_p_matrix = p_matrix[count:count + bt_time_count]
            pair_p_matrix_cur_batch_each_gap = np.zeros(shape=(max_gap, p_matrix_index.shape[0], select_p_matrix.shape[1]), dtype=float)
            for i in range(max_gap):
                p_matrix_index += (direction != "forward") * (-1) if i == 0 else 2 * (direction == "forward") - 1
                valid_mask = (p_matrix_index >= 0) & (p_matrix_index < select_p_matrix.shape[0])
                valid_rows = select_p_matrix[p_matrix_index[valid_mask]]
                pair_p_matrix_cur_batch_each_gap[i][valid_mask] = valid_rows
                pair_p_matrix_cur_batch_each_gap[i][pair_df_cur_batch["gap"] <= i] = 0
            
            pair_p_matrix_cur_batch = np.concatenate(pair_p_matrix_cur_batch_each_gap, axis=1)
            pair_p_matrix = np.row_stack((pair_p_matrix, pair_p_matrix_cur_batch))

            count += bt_time_count

        return pair_df, (pair_gap_mx, pair_o_unit_mx, pair_o_matrix), pair_p_matrix

    # -------------------------------------------------------------------------------------
    @staticmethod
    def _dup_rows(mx, idx, dup_num=1):
        return np.insert(mx, [idx + 1] * dup_num, mx[idx], axis=0)

    @staticmethod
    def _dup_cols(mx, idx, dup_num=1):
        return np.insert(mx, [idx + 1] * dup_num, mx[:, [idx]], axis=1)

    # -------------------------------------------------------------------------------------
    # Other methods
    # -------------------------------------------------------------------------------------
    def __repr__(self):
        return "[{} observations x {} variates x {} perturbations, {} batch in total]".format(
            self.calc_params["obs"], self.calc_params["ovar"], self.calc_params["pvar"], self.calc_params["batch"]
        )

    @property
    def shape(self):
        return self.calc_params["obs"], self.calc_params["ovar"], self.calc_params["pvar"], self.calc_params["batch"]
