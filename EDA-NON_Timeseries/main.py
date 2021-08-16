import pickle
import numpy as np


if __name__ == '__main__':

    heart_disease_detector_model = pickle.load(open('./EDA-NON_Timeseries/models/heart_disease_detector.pickle', 'rb'))
    
    features = ["age","sex",
                "cp","trtbps",
                "chol","fbs",
                "restecg","thalachh",
                "exng","oldpeak",
                "slp","caa","thall"]
    user_input = []
    for item in features:
        input_value = input("Enter the value according to description")
        user_input.append(input_value)
    print(heart_disease_detector_model.predict(np.array(user_input)))

