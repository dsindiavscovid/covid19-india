import os
import numpy as np

from modules.data_fetcher_module import DataFetcherModule

# Fixes the input staff ratio matrix and also the column names, index
def get_clean_staffing_ratio(staff_ratios_file_path: str):
    staff_ratios = DataFetcherModule.get_staffing_ratios(staff_ratios_file_path)
    for b in staff_ratios.columns:
        bnew = b.split('\n')[0].strip()
        staff_ratios = staff_ratios.rename(columns= {b:bnew})
    staff_ratios = staff_ratios.set_index('Personnel')
    staff_ratios.fillna(0,inplace=True)
    return staff_ratios.copy()

# Computes the staffing matrix for a given active_count and other params
def compute_staffing_matrix(active_count: int, bed_type_ratio: int,
                            staff_ratios_file_path: str,
                            bed_multiplier_factor: int):

    # fix the staff ratios  
    staff_df = get_clean_staffing_ratio(staff_ratios_file_path)
    
    # add a beds rows
    staff_df.loc['Beds',:]= bed_multiplier_factor
    
    
    bed_units ={}
    staff_df['Total']=0
    # compute bed units, staff counts and sums
    #(each unit = multiplier_factor beds)
    for bed_type in bed_type_ratio:
        bed_units[bed_type]= bed_type_ratio[bed_type]*active_count/bed_multiplier_factor
        staff_df[bed_type] = staff_df[bed_type]*bed_units[bed_type]
        staff_df['Total'] =staff_df['Total'] + staff_df[bed_type]
  

    # select only relevant columns
    inds = list(bed_units.keys())
    inds = inds + ['Total']
    
    return staff_df[inds]
