# Quick Start Guide
## Introduction
Welcome to LSTS (land surface time sereies)! This guide will help you get started quickly with the installation, configuration, and basic usage of LSTS.
## Prerequisites
- Python 3.8
## Installation
```bash
pip install lsts
```
## First Steps
### Download Checkpoints
```python
from lsts import download_checkpoints

# local_dir is the directory where the checkpoints will be downloaded
download_checkpoints(local_dir="checkpoints")
```
## Common Tasks
### Task 1: Long-term Forecasting
```python
# First, import the LongTermForecast class
from lsts import LongTermForecast

# PRED_LEN = Literal[96, 192, 336, 720]
# MODEL_NAME = Literal["iTransformer", "LSTM", "PatchTST", "DLinear", "TimesNet", "EALSTM"]
# VARIABLE = Literal["air_temperature", "snow_depth", "snow_water_equivalent", "soil_moisture", "soil_suction", "soil_temperature", "surface_temperature"]

# Instantiate the LongTermForecast class
ltf = LongTermForecast(pred_len=96, variable="soil_temperature")

# src_seq is of type pandas.DataFrame and should include timestamp and variable columns. 
# static_variable is of type pandas.Series and only applies when model_name="EALSTM".
ouput_seq = ltf.pred(src_seq, static_variable)

# Visualize the results
# tgt_seq is of type pandas.DataFrame and should include timestamp and variable columns. 
ltf.visual(
    input_seq=src_seq, 
    output_seq=output_seq, 
    variable="soil_temperature", 
    save_path="st_ltf.pdf", 
    ground_truth=tgt_seq
)
```
### Task 2: Short-term Forecasting
```python
# First, import the ShortTermForecast class
from lsts import ShortTermForecast

# MODEL_NAME = Literal["iTransformer", "LSTM", "PatchTST", "DLinear", "TimesNet", "EALSTM"]
# VARIABLE = Literal["air_temperature", "snow_depth", "snow_water_equivalent", "soil_moisture", "soil_suction", "soil_temperature", "surface_temperature"]

# Instantiate the ShortTermForecast class
stf = ShortTermForecast("soil_temperature")

# src_seq is of type pandas.DataFrame and should include timestamp and variable columns. 
# static_variable is of type pandas.Series and only applies when model_name="EALSTM".
ouput_seq = stf.pred(src_seq, static_variable)

# Visualize the results
# tgt_seq is of type pandas.DataFrame and should include timestamp and variable columns. 
stf.visual(
    input_seq=src_seq, 
    output_seq=output_seq, 
    variable="soil_temperature", 
    save_path="st_stf.pdf", 
    ground_truth=tgt_seq
)
```
### Task 3: Imputation (Missing Data)
```python
# First, import the Imputation class
from lsts import Imputation

# Instantiate the Imputation class
imputer = Imputation(
    variable="soil_moisture", 
    model_name="TimesNet"
)
# src_seq is of type pandas.DataFrame and should include timestamp and variable columns. 
# static_variable is of type pandas.Series and only applies when model_name="EALSTM".
output_seq = imputer.pred(src_seq, static_variable)

# Visualize the results
# tgt_seq is of type pandas.DataFrame and should include timestamp and variable columns.
imputer.visual(
    input_seq=src_seq, 
    output_seq=output_seq, 
    variable="soil_moisture", 
    save_path="st_imputation.pdf", 
    ground_truth=tgt_seq
)
```
## Conclusion
You are now ready to start using LSTS. For more detailed instructions and, refer to our paper(under review). If you have any questions, please feel free to contact us.
