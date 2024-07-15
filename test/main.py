from lsts.long_term_forecast import LongTermForecast
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from tqdm import tqdm
from ismn.interface import ISMN_Interface
import os

data_path = "Data_separate_files_header_20240301_20240622_9562_27gu_20240622.zip"
ismn_data = ISMN_Interface(data_path, parallel=False)
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

variables = ["air_temperature", "snow_depth", "snow_water_equivalent", "soil_moisture", "soil_suction", "soil_temperature", "surface_temperature"]
for variable in variables:
    logs = []
    ltf = LongTermForecast(pred_len=96, variable=variable, model_name="TimesNet")
    pbar = tqdm(total=len(ismn_data.metadata[ismn_data.metadata['variable'].val == variable]))
    for network, station, sensor in ismn_data.collection.iter_sensors(variable=variable):
        pbar.update()
        data: pd.DataFrame = sensor.data
        data.reset_index(inplace=True, drop=False)
        data.loc[data[f"{variable}_flag"] != "G", variable] = np.nan
        data = data.rename(columns={"date_time": "date"})[["date", variable]].copy()
        start = 0
        counts = 0
        mse_lst = []
        mae_lst = []
        while start < len(data):
            end = start + 512
            if end + 96 >= len(data):
                break
            if data.iloc[start: end + 96][variable].isna().sum() > 0:
                temp: pd.DataFrame = data.iloc[start: end + 96].reset_index(drop=True)
                next_index = temp[variable].isna().idxmax()
                start += (next_index + 1)
                continue
            if data.iloc[start: end][variable].std() == 0:
                start = start + end
                continue
            pred = ltf.pred(data.iloc[start: end].copy())
            counts += 1
            true = data[end: end + 96]

            mse_val = mean_squared_error(true[variable], pred[variable], squared=False)
            mae_val = mean_absolute_error(true[variable], pred[variable])
            mse_lst.append(mse_val)
            mae_lst.append(mae_val)
            start += 1
            # comments
            # if counts < 10:
            #     print(mse_val, mae_val, variable)
            # else:
            #     break
        if counts == 0:
            continue
        logs.append(
            pd.DataFrame({"lon": sensor.metadata.to_pd().longitude.val, "lat": sensor.metadata.to_pd().latitude.val, 
                        "rmse_val": sum(mse_lst) / len(mse_lst), "mae_val": sum(mae_lst) / len(mae_lst), "counts": counts}, index=[0])
        )
        # comments
        # break
    if len(logs) == 0:
        continue
    df_logs = pd.concat(logs, axis=0)
    df_logs.to_csv(os.path.join(save_dir, f"{variable}.csv"), index=False)
    pbar.close()