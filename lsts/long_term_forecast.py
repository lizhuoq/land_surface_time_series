import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

from typing import Literal, Tuple, Optional, List
from os import listdir
from os.path import join, dirname
from joblib import load
import warnings

from .models import LSTM, iTransformer, PatchTST, DLinear, TimesNet, EALSTM
from .utils.timefeatures import time_features

MODEL_NAME = Literal["iTransformer", "LSTM", "PatchTST", "DLinear", "TimesNet", "EALSTM"]
VARIABLE = Literal["air_temperature", "precipitation", "snow_depth", "snow_water_equivalent", "soil_moisture", "soil_suction", "soil_temperature", "surface_temperature"]
# STRATEGY = Literal["single", "ensemble"]
PRED_LEN = Literal[96, 192, 336, 720]
__CURRENT_DIR = dirname(__file__)
# CHECKPOINT_ROOT = join(__CURRENT_DIR, "checkpoints")
SCALER_ROOT = join(__CURRENT_DIR, "scaler")

warnings.filterwarnings("ignore", message="Trying to unpickle estimator StandardScaler from version .* when using version .*")
warnings.filterwarnings("ignore", message="X does not have valid feature names, but StandardScaler was fitted with feature names")


class LongTermForecast:
    def __init__(self, 
                 pred_len: PRED_LEN,  
                 variable: VARIABLE, 
                 checkpoints_dir: Optional[str] = None, 
                 model_name: MODEL_NAME = "iTransformer") -> None:
        self.model_dict = {
            "LSTM": LSTM, 
            "iTransformer": iTransformer, 
            "PatchTST": PatchTST, 
            "DLinear": DLinear, 
            "TimesNet": TimesNet, 
            "EALSTM": EALSTM
        }
        # freq "h"
        self.checkpoints_dir = "checkpoints" if checkpoints_dir is None else checkpoints_dir
        self.variable = variable
        self.pred_len = pred_len
        self.model_name = model_name
        self.model = self.__load_checkpoint(self._build_model())
        self.scaler = self.__load_scaler()

    def __load_checkpoint(self, model: nn.Module) -> nn.Module:
        checkpoint_path = [x for x in listdir(self.checkpoints_dir) 
                           if x.startswith(f"long_term_forecast_ismn_{self.variable}_512_{self.pred_len}_{self.model_name}")]
        assert len(checkpoint_path) == 1
        checkpoint_path = checkpoint_path[0]
        checkpoint_path = join(self.checkpoints_dir, checkpoint_path, "checkpoint.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        return model
    
    def __load_scaler(self) -> StandardScaler:
        scaler_path = join(SCALER_ROOT, f"long_term_forecast_{self.variable}.joblib")
        scaler = load(scaler_path)
        return scaler
        
    def _get_lstm_configs(self, pred_len: PRED_LEN):
        class Args():
            def __init__(self) -> None:
                self.task_name = "long_term_forecast"
                self.seq_len = 512
                self.pred_len = pred_len
                self.enc_in = 1
                self.d_model = 16
                self.e_layers = 3
                self.dropout = 0.2
                self.bidirectional = False
                self.c_out = 1
        args = Args()
        return args
    
    def _get_timesnet_configs(self, pred_len: PRED_LEN):
        class Args():
            def __init__(self) -> None:
                self.seq_len = 512
                self.pred_len = pred_len
                self.top_k = 5
                self.d_model = 32
                self.d_ff = 32
                self.num_kernels = 6
                self.task_name = "long_term_forecast"
                self.label_len = 512 // 2
                self.e_layers = 2
                self.enc_in = 1
                self.embed = "timeF"
                self.freq = "h"
                self.dropout = 0.1
                self.c_out = 1
        args = Args()
        return args
    
    def _get_patchtst_configs(self, pred_len: PRED_LEN):
        class Args():
            def __init__(self) -> None:
                self.task_name = "long_term_forecast"
                self.seq_len = 512
                self.pred_len = pred_len
                self.d_model = 16
                self.dropout = 0.2
                self.factor = 3
                self.output_attention = False
                self.n_heads = 4
                self.d_ff = 128
                self.activation = "gelu"
                self.e_layers = 3
                self.enc_in = 1
        args = Args()
        return args
    
    def _get_dlinear_configs(self, pred_len: PRED_LEN):
        class Args():
            def __init__(self) -> None:
                self.task_name = "long_term_forecast"
                self.seq_len = 512
                self.pred_len = pred_len
                self.moving_avg = 25
                self.enc_in = 1
        args = Args()
        return args
    
    def _get_itransformer_configs(self, pred_len: PRED_LEN):
        class Args():
            def __init__(self) -> None:
                self.task_name = "long_term_forecast"
                self.seq_len = 512
                self.pred_len = pred_len
                self.output_attention = False
                self.d_model = 512
                self.embed = "timeF"
                self.freq = "h"
                self.dropout = 0.1
                self.factor = 3
                self.n_heads = 8
                self.d_ff = 512
                self.activation = "gelu"
                self.e_layers = 3
        args = Args()
        return args
    
    def _get_ealstm_configs(self, pred_len: PRED_LEN):
        args = self._get_lstm_configs(pred_len)
        args.numeric_stat_in = 5 if self.variable in ["precipitation", "air_temperature"] else 10
        args.d_model = 256
        args.dropout = 0.4
        return args

    def _build_model(self) -> nn.Module:
        if self.model_name == "LSTM":
            model = self.model_dict[self.model_name].Model(self._get_lstm_configs(self.pred_len)).float()
        elif self.model_name == "TimesNet":
            model = self.model_dict[self.model_name].Model(self._get_timesnet_configs(self.pred_len)).float()
        elif self.model_name == "PatchTST":
            model = self.model_dict[self.model_name].Model(self._get_patchtst_configs(self.pred_len)).float()
        elif self.model_name == "DLinear":
            model = self.model_dict[self.model_name].Model(self._get_dlinear_configs(self.pred_len)).float()
        elif self.model_name == "iTransformer":
            model = self.model_dict[self.model_name].Model(self._get_itransformer_configs(self.pred_len)).float()
        elif self.model_name == "EALSTM":
            model = self.model_dict[self.model_name].Model(self._get_ealstm_configs(self.pred_len)).float()
        return model
    
    def pred(self, input_seq: pd.DataFrame, static_variable: Optional[dict] = None) -> pd.DataFrame:
        if self.model_name == "EALSTM":
            return self.pred_ealstm(input_seq, static_variable)
        # [date, variable]
        self.model.eval()
        x_enc, x_mark_enc = self.preprocess(input_seq)
        output = self.model(x_enc, x_mark_enc, None, None).squeeze(0).detach().numpy() # [T,C]
        output = self.inverse_transforme(output)[:, 0]

        input_seq["date"] = pd.to_datetime(input_seq["date"])
        start_time = input_seq["date"].max() + pd.Timedelta(hours=1)
        time_series = pd.date_range(start=start_time, periods=self.pred_len, freq="h")
        df_output = pd.DataFrame({"date": time_series, self.variable: output})
        return df_output
    
    def pred_ealstm(self, input_seq: pd.DataFrame, static_variable: pd.Series) -> pd.DataFrame:
        if static_variable is None:
            raise ValueError("static_variable cannot be None.")
        self.model.eval()
        x_enc, x_mark_enc = self.preprocess(input_seq)
        numeric_s, climate_c, lc_s = self.preprocess_static(static_variable)
        output = self.model(x_enc, numeric_s, climate_c, lc_s).squeeze(0).detach().numpy()
        output = self.inverse_transforme(output)[:, 0]

        input_seq["date"] = pd.to_datetime(input_seq["date"])
        start_time = input_seq["date"].max() + pd.Timedelta(hours=1)
        time_series = pd.date_range(start=start_time, periods=self.pred_len, freq="h")
        df_output = pd.DataFrame({"date": time_series, self.variable: output})
        return df_output

    def preprocess_static(self, static_variable: pd.Series) -> Tuple[torch.Tensor]:
        if self.variable in ["precipitation", "air_temperature"]:
            numeric_columns = [("elevation", "val"), ("latitude", "val"), ("longitude", "val"), 
                                ("variable", "depth_from"), ("variable", "depth_to")]
        else:
            numeric_columns = [("elevation", "val"), ("latitude", "val"), ("longitude", "val"), 
                                ("variable", "depth_from"), ("variable", "depth_to"), ("clay_fraction", "val"), 
                                ("organic_carbon", "val"), ("sand_fraction", "val"), ("saturation", "val"), 
                                ("silt_fraction", "val")]
        climate_KG = ['Af', 'Am', 'Aw', 'BSh', 'BSk', 'BWh', 'BWk', 'Cfa', 'Cfb', 'Cfc', 'Csa', 'Csb', 'Cwa', 
                      'Cwb', 'Dfa', 'Dfb', 'Dfc', 'Dfd', 'Dsa', 'Dsb', 'Dsc', 'Dwa', 'Dwb', 'Dwc', 'EF', 'ET']
        lc = [10, 11, 12, 20, 30, 40, 50, 60, 61, 62, 70, 80, 90, 100, 110, 120, 130, 140, 150, 152, 160, 180, 
              190, 200, 201, 210, 220]
        climate_KG_dict = {x: i for i, x in enumerate(climate_KG)}
        lc_dict = {x: i for i, x in enumerate(lc)}

        numeric_s = static_variable[numeric_columns].values
        numeric_s = numeric_s.astype(np.float32)
        climate_s = np.array([climate_KG_dict[static_variable[("climate_KG", "val")]]])
        lc_s = np.array([lc_dict[static_variable[("lc_2000", "val")]], 
                         lc_dict[static_variable[("lc_2005", "val")]], 
                         lc_dict[static_variable[("lc_2010", "val")]]])
        return torch.tensor(numeric_s).float().unsqueeze(0), torch.tensor(climate_s).long().unsqueeze(0), torch.tensor(lc_s).long().unsqueeze(0)

    def preprocess(self, input_seq: pd.DataFrame) -> Tuple[torch.Tensor]:
        self.check_dataframe_length(input_seq, 512)
        self.check_time_continuity(input_seq)
        input_seq["date"] = pd.to_datetime(input_seq["date"])
        input_seq.sort_values("date", ascending=True, inplace=True)

        data = input_seq[[self.variable]][-512:]
        data = self.scaler.transform(data.values)

        df_stamp = input_seq[["date"]][-512:]
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq="h")
        data_stamp = data_stamp.transpose(1, 0)

        data = data.astype(np.float32)
        data_stamp = data_stamp.astype(np.float32)
        return torch.tensor(data).unsqueeze(0), torch.tensor(data_stamp).unsqueeze(0) # [B,T,C]
    
    def inverse_transforme(self, data):
        return self.scaler.inverse_transform(data)
    
    @staticmethod
    def check_dataframe_length(input_seq: pd.DataFrame, seq_len: int):
        length = len(input_seq)
        if length < seq_len:
            raise ValueError(f"DataFrame length must be at least {input_seq}")
        
    @staticmethod
    def check_time_continuity(input_seq: pd.DataFrame):
        if "date" not in input_seq.columns.to_list():
            raise ValueError("date column not found in DataFrame")
        
        input_seq["date"] = pd.to_datetime(input_seq["date"])
        min_time = input_seq["date"].min()
        max_time = input_seq["date"].max()
        if max_time - min_time != pd.Timedelta(f"{(len(input_seq) - 1)}h"):
            raise ValueError("date column contains non-continuous values")

    @staticmethod
    def visual(input_seq: pd.DataFrame, output_seq: pd.DataFrame, variable: VARIABLE, save_path: str, 
               ground_truth: Optional[pd.DataFrame] = None):
        input_seq["date"] = pd.to_datetime(input_seq["date"])
        input_seq.sort_values("date", ascending=True, inplace=True)

        if ground_truth is not None:
            ground_truth["date"] = pd.to_datetime(ground_truth["date"])
            ground_truth.sort_values("date", ascending=True, inplace=True)

        fig, ax = plt.subplots(1, 1)
        input_seq.set_index("date")[[variable]].rename(columns={variable: "Input Sequence"}).plot(ax=ax)
        output_seq.set_index("date")[[variable]].rename(columns={variable: "Prediction"}).plot(ax=ax, marker="^", linestyle="-.", 
                                                                                               markersize=4, linewidth=1)
        if ground_truth is not None:
            ground_truth.set_index("date")[[variable]].rename(columns={variable: "GroundTruth"}).plot(ax=ax, marker="o", linestyle="-.", 
                                                                                                      markersize=4, linewidth=1)

        plt.savefig(save_path, bbox_inches="tight")

    # def finetune(self, time_series: pd.DataFrame | List[pd.DataFrame]):
    #     if isinstance(time_series, pd.DataFrame):
    #         self.check_dataframe_length(time_series, 512 + self.pred_len)
    #         self.check_time_continuity(time_series)
    #     elif isinstance(time_series, list):
    #         for data in time_series:
    #             self.check_dataframe_length(data, 512 + self.pred_len)
    #             self.check_time_continuity(data)