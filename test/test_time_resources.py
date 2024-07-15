from ismn.interface import ISMN_Interface
from lsts.long_term_forecast import LongTermForecast
import pandas as pd
import time
from tqdm import tqdm
import psutil
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pred_len', type=int, required=True)
args = parser.parse_args()

data_path = "Data_separate_files_header_20240301_20240622_9562_27gu_20240622.zip"
ismn_data = ISMN_Interface(data_path, parallel=False)
for network, station, sensor in ismn_data.collection.iter_sensors(variable='soil_moisture',
                                                                  depth=[0., 0.05]):
    data = sensor.data
    metadata = sensor.metadata.to_pd()
    data = data.reset_index().rename(columns={"date_time": "date"})
    data = data[['date', 'soil_moisture']].copy()
    static_data = metadata.copy()
    break


def pred(data: pd.DataFrame, static_data: dict, model_name, pred_len=args.pred_len):
    predicter = LongTermForecast(pred_len=pred_len, variable="soil_moisture", model_name=model_name)
    predicter.pred(data, static_data)


pid = os.getpid()
p = psutil.Process(pid)
log_times = []
log_memorys = []

for _ in tqdm(range(1)):
    start_memory = p.memory_info().rss
    start_time = time.time()
    pred(data, static_data, "TimesNet")
    end_time = time.time()
    end_memory = p.memory_info().rss

    elapsed_time = end_time - start_time
    memory_used = end_memory - start_memory

    log_times.append(elapsed_time)
    log_memorys.append(memory_used / (1024 * 1024))

df = pd.DataFrame({"time": log_times, "memory": log_memorys, 'pred_len': [args.pred_len], 'model': ["TimesNet"]})
if os.path.exists("resources.csv"):
    df_log = pd.read_csv("resources.csv")
    df_log = pd.concat([df_log, df])
else:
    df_log = df
df_log.to_csv("resources.csv", index=False)