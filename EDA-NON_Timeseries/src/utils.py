
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score

def histogram_plot(dataframe, target : str, feature : str):
    have_disease = dataframe[dataframe[target]==1][feature].value_counts()
    no_disease  = dataframe[dataframe[target]==0][feature].value_counts()
    df = pd.DataFrame([have_disease, no_disease])
    df.index = ['Have disease','No disease']
    df.plot(kind='bar',stacked = False, title=str(feature))

def numerical_data_analysis(dataframe, feature : list):
    plt.hist(dataframe[dataframe['output']==1][feature],
            color='red',
            edgecolor='yellow',
            alpha=0.4,
            label=f'{feature} - have disease')

    plt.hist(dataframe[dataframe['output']==0][feature],
            color='black',
            edgecolor='blue',
            alpha=0.4,
            label=f'{feature} - dont have disease')

    
    plt.legend(loc='upper left')

def remove_outliers(dataframe):
    zscore = np.abs(stats.zscore(dataframe))
    ouliers_free_data = dataframe[(zscore<3).all(axis=1)]
    return pd.DataFrame(ouliers_free_data)

def plot_boxplot(dataframe, features : list):
    plt.figure(figsize=(20,15))
    
    for index, feature in enumerate(features):
    
        plt.subplot(4,4,index+1)
        sns.boxplot(y = feature, data = dataframe)

def datasplit(dataframe, test_size : float ):
    split = StratifiedShuffleSplit(n_splits = 1, test_size = test_size, random_state=0)
    for train_index, test_index in split.split(dataframe, dataframe['sex']):
        train_set = dataframe.loc[train_index]
        test_set = dataframe.loc[test_index]
    return train_set, test_set


def model_building(model, x_train, y_train):

    model.fit(x_train,y_train)
    train_score = model.score(x_train , y_train)
    shuffle_split = StratifiedShuffleSplit(train_size=0.8, 
                                            test_size=0.2, 
                                            n_splits=5, 
                                            random_state=0)
    val_score = cross_val_score(model, x_train , y_train, cv = shuffle_split)
    return train_score, np.mean(val_score)

def model_evaluation(model, x_test, y_test):
    result_dict = {
    "precision_score" : precision_score(y_test, model.predict(x_test)),
    "recall_score" : recall_score(y_test, model.predict(x_test)),
    "f1_score" : f1_score(y_test, model.predict(x_test))   
    } 
    return result_dict



