"""

"""
import pandas as pd 
import numpy as np 

"""
function = concat_dataset
input = list 
output = pandas dataframe 
"""
def concat_dataset(company_list : list, path : str ):
    concat_data = pd.DataFrame()

    for file in company_list:
        current_df = pd.read_csv(path +"/"+file)
        concat_data = pd.concat([concat_data, current_df])
    
    return(concat_data)



