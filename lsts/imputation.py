from .long_term_forecast import *

__CURRENT_DIR = dirname(__file__)
# CHECKPOINT_ROOT = join(__CURRENT_DIR, "checkpoints")
SCALER_ROOT = join(__CURRENT_DIR, "scaler")


class Imputation(LongTermForecast):
    def __init__(self, 
                 variable: VARIABLE, 
                 model_name: MODEL_NAME = "TimesNet", 
                 checkpoints_dir: Optional[str] = None) -> None:
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
        self.model_name = model_name
        self.task_name = "imputation"
        self.pred_len = 0
        self.model = self.__load_checkpoint(self._build_model())
        self.scaler = self.__load_scaler()

    def __load_checkpoint(self, model: nn.Module) -> nn.Module:
        checkpoint_path = [x for x in listdir(self.checkpoints_dir) 
                           if x.startswith(f"imputation_ismn_{self.variable}_mask_0.5_{self.model_name}")]
        assert len(checkpoint_path) == 1
        checkpoint_path = checkpoint_path[0]
        checkpoint_path = join(self.checkpoints_dir, checkpoint_path, "checkpoint.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        return model
    
    def __load_scaler(self) -> StandardScaler:
        scaler_path = join(SCALER_ROOT, f"imputation_short_term_forecast_{self.variable}.joblib")
        scaler = load(scaler_path)
        return scaler
    
    def _get_timesnet_configs(self, pred_len: int):
        args = super()._get_timesnet_configs(pred_len)
        args.seq_len = 96
        args.pred_len = 0
        args.top_k = 3
        args.d_model = 64
        args.d_ff = 64
        args.task_name = self.task_name
        args.label_len = 0
        return args
    
    def _get_lstm_configs(self, pred_len: int):
        args = super()._get_lstm_configs(pred_len)
        args.task_name = self.task_name
        args.seq_len = 96
        args.pred_len = 0
        return args
    
    def _get_ealstm_configs(self, pred_len: int):
        args = super()._get_ealstm_configs(pred_len)
        args.task_name = self.task_name
        args.seq_len = 96
        args.pred_len = 0
        return args

    def _get_patchtst_configs(self, pred_len: int):
        args = super()._get_patchtst_configs(pred_len)
        args.task_name = self.task_name
        args.seq_len = 96
        args.pred_len = 0
        return args
    
    def _get_dlinear_configs(self, pred_len: int):
        args = super()._get_dlinear_configs(pred_len)
        args.task_name = self.task_name
        args.seq_len = 96
        args.pred_len = 0
        return args
    
    def _get_itransformer_configs(self, pred_len: int):
        args = super()._get_itransformer_configs(pred_len)
        args.task_name = self.task_name
        args.seq_len = 96
        args.pred_len = 0
        return args
    
    def pred(self, input_seq: pd.DataFrame, static_variable: pd.Series) -> pd.DataFrame:
        if self.model_name == "EALSTM":
            return self.pred_ealstm(input_seq, static_variable)
        self.model.eval()
        x_enc, x_mark_enc, mask = self.preprocess(input_seq)
        output = self.model(x_enc, x_mark_enc, None, None, mask).squeeze(0).detach().numpy()
        output = self.inverse_transforme(output)[:, 0]

        mask = mask.squeeze(0).numpy()[:, 0]
        x_enc = self.inverse_transforme(x_enc.squeeze(0).numpy())[:, 0]
        output = np.where(mask==0, output, x_enc)

        input_seq["date"] = pd.to_datetime(input_seq["date"])
        date = input_seq["date"].sort_values(ascending=True)
        df_output = pd.DataFrame({
            "date": date, self.variable: output
        })
        return df_output
    
    def pred_ealstm(self, input_seq: pd.DataFrame, static_variable: pd.Series) -> pd.DataFrame:
        if static_variable is None:
            raise ValueError("static_variable cannot be None.")
        self.model.eval()
        x_enc, x_mark_enc, mask = self.preprocess(input_seq)
        numeric_s, climate_c, lc_s = self.preprocess_static(static_variable)
        output = self.model(x_enc, numeric_s, climate_c, lc_s).squeeze(0).detach().numpy()
        output = self.inverse_transforme(output)[:, 0]

        mask = mask.squeeze(0).numpy()[:, 0]
        x_enc = self.inverse_transforme(x_enc.squeeze(0).numpy())[:, 0]
        output = np.where(mask==0, output, x_enc)

        input_seq["date"] = pd.to_datetime(input_seq["date"])
        date = input_seq["date"].sort_values(ascending=True)
        df_output = pd.DataFrame({
            "date": date, self.variable: output
        })
        return df_output

    def preprocess(self, input_seq: pd.DataFrame):
        self.check_dataframe_length(input_seq)
        self.check_mask_rate(input_seq)
        self.check_time_continuity(input_seq)

        input_seq["date"] = pd.to_datetime(input_seq["date"])
        input_seq.sort_values("date", ascending=True, inplace=True)

        mask = input_seq[["is_mask"]].copy()
        mask["is_mask"] = mask["is_mask"].apply(lambda x: abs(x - 1))
        mask = mask.values

        data = input_seq[[self.variable, "is_mask"]].copy()
        mask_index = data[data["is_mask"] == 1].index
        data.loc[mask_index, self.variable] = data[self.variable].mean()
        data = self.scaler.transform(data[[self.variable]].values).copy()
        data = np.where(mask==0, 0, data)

        df_stamp = input_seq[["date"]]
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq="h")
        data_stamp = data_stamp.transpose(1, 0)

        data = data.astype(np.float32)
        data_stamp = data_stamp.astype(np.float32)
        mask = mask.astype(np.float32)
        return torch.tensor(data).unsqueeze(0), torch.tensor(data_stamp).unsqueeze(0), torch.tensor(mask).unsqueeze(0) # [B,T,C]

    def check_mask_rate(self, input_seq: pd.DataFrame):
        '''
        date, variable, is_mask
        '''
        mask_rate = input_seq["is_mask"].sum() / len(input_seq)
        if mask_rate > 0.5:
            raise ValueError(f"The proportion of missing values in the dataframe \
                             should be less than or equal to 0.5, but your \
                             proportion of missing values is {mask_rate}.")
        
    def check_dataframe_length(self, input_seq: pd.DataFrame):
        length = len(input_seq)
        if length != 96:
            raise ValueError(f"The length of the dataframe should be equal to 96.")
        