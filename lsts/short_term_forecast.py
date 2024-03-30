from .long_term_forecast import *

__CURRENT_DIR = dirname(__file__)
# CHECKPOINT_ROOT = join(__CURRENT_DIR, "checkpoints")
SCALER_ROOT = join(__CURRENT_DIR, "scaler")


class ShortTermForecast(LongTermForecast):
    def __init__(self, variable: VARIABLE, model_name: MODEL_NAME = "iTransformer", checkpoints_dir: Optional[str] = None) -> None:
        self.model_dict = {
            "LSTM": LSTM, 
            "iTransformer": iTransformer, 
            "PatchTST": PatchTST, 
            "DLinear": DLinear, 
            "TimesNet": TimesNet, 
            "EALSTM": EALSTM
        }
        self.checkpoints_dir = "checkpoints" if checkpoints_dir is None else checkpoints_dir
        self.variable = variable
        self.pred_len = 48
        self.model_name = model_name
        self.task_name = "short_term_forecast"
        self.model = self.__load_checkpoint(self._build_model())
        self.scaler = self.__load_scaler()      

    def __load_checkpoint(self, model: nn.Module) -> nn.Module:
        checkpoint_path = [x for x in listdir(self.checkpoints_dir) 
                           if x.startswith(f"short_term_forecast_ismn_{self.variable}_48_48_{self.model_name}")]
        assert len(checkpoint_path) == 1
        checkpoint_path = checkpoint_path[0]
        checkpoint_path = join(self.checkpoints_dir, checkpoint_path, "checkpoint.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        return model
    
    def __load_scaler(self) -> StandardScaler:
        scaler_path = join(SCALER_ROOT, f"imputation_short_term_forecast_{self.variable}.joblib")
        scaler = load(scaler_path)
        return scaler
    
    def _get_lstm_configs(self, pred_len: int):
        args = super()._get_lstm_configs(pred_len)
        args.task_name = self.task_name
        args.seq_len = 48
        return args
    
    def _get_timesnet_configs(self, pred_len: int):
        args = super()._get_timesnet_configs(pred_len)
        args.seq_len = 48
        args.d_model = 16
        args.d_ff = 16
        args.task_name = self.task_name
        args.label_len = 48 // 2
        return args
    
    def _get_patchtst_configs(self, pred_len: int):
        args = super()._get_patchtst_configs(pred_len)
        args.task_name = self.task_name
        args.seq_len = 48
        return args
    
    def _get_dlinear_configs(self, pred_len: int):
        args = super()._get_dlinear_configs(pred_len)
        args.task_name = self.task_name
        args.seq_len = 48
        return args
    
    def _get_itransformer_configs(self, pred_len: int):
        args = super()._get_itransformer_configs(pred_len)
        args.task_name = self.task_name
        args.seq_len = 48
        return args
    
    def _get_ealstm_configs(self, pred_len: int):
        args = super()._get_ealstm_configs(pred_len)
        args.task_name = self.task_name
        args.seq_len = 48
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
    
    def preprocess(self, input_seq: pd.DataFrame) -> Tuple[torch.Tensor]:
        self.check_dataframe_length(input_seq, 48)
        self.check_time_continuity(input_seq)
        input_seq["date"] = pd.to_datetime(input_seq["date"])
        input_seq.sort_values("date", ascending=True, inplace=True)

        data = input_seq[[self.variable]][-48:]
        data = self.scaler.transform(data.values)

        df_stamp = input_seq[["date"]][-48:]
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq="h")
        data_stamp = data_stamp.transpose(1, 0)

        data = data.astype(np.float32)
        data_stamp = data_stamp.astype(np.float32)
        return torch.tensor(data).unsqueeze(0), torch.tensor(data_stamp).unsqueeze(0) # [B,T,C]
    