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

def figureplot(dataframe, x_axis ,y_axis, choose_plot):
    tech_list = dataframe['Name'].unique()
    if x_axis == 'date':
        dataframe[x_axis] = pd.to_datetime(dataframe[x_axis])
        if(choose_plot == 'matplot'):

            plt.figure(figsize=(20,12))
            for index, company in enumerate(tech_list,1):
                plt.subplot(2, 3, index)
                df=dataframe[dataframe['Name']==company]
                figure = plt.plot(df[x_axis],df[y_axis])
                company_title = plt.title(company)

            return figure, company_title

        elif(choose_plot == 'plotly'):
            for company in (tech_list):
                figure = px.line(dataframe, x = x_axis, y = y_axis, title=company)
            return figure
    
def calc_daily_change(dataframe):
    dataframe['daily%_price_change'] = ((dataframe['close'] - dataframe['open'])/dataframe['close'])*100
    return dataframe

def make_dataframe(dataframe, feature : str):
    tech_list = dataframe['Name'].unique()
    close_price = pd.DataFrame()
    for company in tech_list:
        df=dataframe[dataframe['Name']==company]
        close_price[company]=df[feature]
    return close_price
