import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == "imputation":
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.lstm = nn.LSTM(input_size=self.channels, 
                            hidden_size=configs.d_model, 
                            num_layers=configs.e_layers, 
                            batch_first=True, 
                            dropout=configs.dropout, 
                            bidirectional=configs.bidirectional)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len)
        h_out = 2 * configs.d_model if configs.bidirectional else configs.d_model
        self.projection = nn.Linear(h_out, configs.c_out)

    def encoder(self, x_enc):
        enc_out = self.lstm(x_enc)[0] # [B, L, D]
        output = self.predict_linear(enc_out.transpose(1, 2)).transpose(1, 2)
        output = self.projection(output)
        return output
    
    def forecast(self, x_enc):
        return self.encoder(x_enc)
    
    def imputation(self, x_enc):
        return self.encoder(x_enc)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "long_term_forecast" or self.task_name == "short_term_forecast":
            dec_out = self.forecast(x_enc)
            return dec_out
        if self.task_name == "imputation":
            dec_out = self.imputation(x_enc)
            return dec_out
        return None
    