# imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Loading in dataset - could consider making this less annoying but testing one for now
ticker = "BB.TO"
dataset = pd.read_csv(f'{ticker}_unadjusted_prices_2007_2018.csv', skiprows=2)
dataset.rename(columns={"Unnamed: 1": "Price"}, inplace=True)
# print(dataset)

# normalizing the data
dataset["Price"] = np.log(dataset["Price"]) # I doubt anything is 0....

# Setting up lags -  efficently so computer doesn't die
lags = range(1,120) # will look at past 120 days

lagged = pd.concat(
    [dataset["Price"].shift(lag) for lag in lags], # create a lag number of values for every price
    axis=1
)
lagged.columns = [f"lag_{lag}" for lag in lags]
dataset = pd.concat([dataset, lagged], axis=1) # connect back to og dataset

# Remove rows with missing lag values
rf_input = dataset.dropna().reset_index(drop=True)

# Writing to a CSV
rf_input.to_csv(f'./RFInputs/{ticker}_rf_input.csv', index=False)

# To verify this will give:
#       Columns: Current date, Lag for day (for 120)
#       Rows: The days we are looking at in total (That have lags - so our first row is only starting at 120 date)b