from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

boston_dataset= load_boston()

data=pd.DataFrame(data=boston_dataset.data,columns=boston_dataset.feature_names)

features=data.drop(['INDUS','AGE'],axis=1)

log_prices=np.log(boston_dataset.target)
target=pd.DataFrame(log_prices,columns=['PRICE'])

property_stats=np.ndarray(shape=(1,11))

property_stats=features.mean().values.reshape(1,11)

regr=LinearRegression()
regr.fit(features,target)
fitted_vals=regr.predict(features)


MSE=mean_squared_error(target,fitted_vals)
RMSE=np.sqrt(MSE)
RM_IDX=4
PTRATIO_IDX=8
CHAS_IDX=2
def get_log_estimate(nr_rooms,students_per_classroom,
                     next_to_river=False,high_confidence=True):
    
    property_stats[0][RM_IDX]=nr_rooms
    property_stats[0][PTRATIO_IDX]=students_per_classroom
    
    log_estimate=regr.predict(property_stats)[0][0]
    
    if next_to_river:
        property_stats[0][CHAS_IDX]=1
    else:
        property_stats[0][CHAS_IDX]=0
    
    if high_confidence:
        upper_bound=log_estimate + 2*RMSE
        lower_bound=log_estimate - 2*RMSE
        interval=95
    else:
        upper_bound=log_estimate + RMSE
        lower_bound=log_estimate - RMSE
        interval=68
    
    return log_estimate ,upper_bound,lower_bound,interval



np.median(boston_dataset.target)

ZILLOW_MEDIAN_PRICE=583.3
SCALE_FACTOR=ZILLOW_MEDIAN_PRICE / np.median(boston_dataset.target)
SCALE_FACTOR

def get_dollar_estimate(rooms,ptratio,river,conf):
    
    
    """
    Estimates the price of apartment in Boston
    
    Keyword Arguements:
    
    rooms-No of rooms in apartment
    ptratio-Teacher Child ratio
    river-Situated alongside river or not
    conf- 68%-95% confidence level
    """
    if rooms < 1 or ptratio < 1:
        print('Unrealistic value.Please check the data')
    else:

        log_est,upper,lower,conf=get_log_estimate(nr_rooms=rooms,students_per_classroom=ptratio,
                                                  next_to_river=river,high_confidence=conf)
        dollar_est=np.e**(log_est)*1000*SCALE_FACTOR
        dollar_low=np.e**(lower)*1000*SCALE_FACTOR
        dollar_high=np.e**(upper)*1000*SCALE_FACTOR

        rounded_est=round(dollar_est,3)
        rounded_low=round(dollar_low,3)
        rounded_high=round(dollar_high,3)

        print(f'The estimated actual price of the house is {rounded_est}.')
        print(f'The estimated lower price of the house is {rounded_low} at {conf}% confidence.')
        print(f'The estimated higher price of the house is {rounded_high} at {conf}% confidence.')
