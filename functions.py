"""

"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import plotly.express as px

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

def figureplot(dataframe, x_axis ,y_aixs, choose_plot):
    if x_axis == 'date':
        dataframe[x_axis] = pd.to_datetime(dataframe[x_axis])
    if choose_plot == "matplot":
        plt.figure(figsize=(20,12))
        tech_list = dataframe['Name'].unique()
        for index, company in enumerate(tech_list,1):
            plt.subplot(2, 3, index)
            df=dataframe[dataframe['Name']==company]
            figure = plt.plot(df[x_axis],df[y_aixs])
            company_title = plt.title(company)

        return figure, company_title

    elif(choose_plot == 'plotly'):
          
        return  (px.line(dataframe, x = x_axis, y = y_aixs, title = ""))
    

       

