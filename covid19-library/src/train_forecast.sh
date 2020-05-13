#!/bin/bash
##usage: ./train_forecast.sh <forecast_end_date(m/d/y)>

# regions=("bengaluru urban" "jaipur" "delhi" "mumbai" "pune" "ahmedabad")
# region_types=("district" "district" "state" "district" "district" "district")

regions=("pune")
region_types=("district")

rlen=${#regions[@]}

for(( i=0; i<${rlen}; i++ ));
        do python train_eval_plot.py --region "${regions[$i]}" --region_type "${region_types[$i]}" --forecast_end_date "$1";
done
