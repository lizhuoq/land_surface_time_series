#!/bin/bash

for i in {1..100}
do
    for pred_len in 96 192 336 720
    do
        python test_time_resources.py --pred_len $pred_len
    done
done