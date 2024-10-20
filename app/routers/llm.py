from fastapi import APIRouter, Depends
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

import pandas as pd

from app.dependencies.kaggle import KaggleClient
router = APIRouter(
    prefix="/llm",
    tags=["llm"],
    dependencies=[Depends(KaggleClient)],
    responses={404: {"llm": "Not found"}},
)



'''
The steps to building and using a model are:
    Define: What type of model will it be? A decision tree? Some other type of model?
    Fit: Capture patterns from provided data. This is the heart of modeling.
    Predict: Just what it sounds like
    Evaluate: Determine how accurate the model's predictions are.
'''
@router.get("/")
def read_items() -> dict:
    path:str = KaggleClient().dataset_download("dansbecker/melbourne-housing-snapshot")
    data = pd.read_csv(path+"/melb_data.csv") 


    # Clean the data
    cleaned_data: pd.DataFrame = data.dropna(axis=0)
    # Define features
    X: pd.DataFrame = cleaned_data[["Rooms", "Bathroom", "Landsize", "BuildingArea", "YearBuilt", "Lattitude", "Longtitude"]]
    # Define target
    y: pd.Series = cleaned_data["Price"]
    # Define model
    model = DecisionTreeRegressor(random_state=1)
    # Train (or Fit) the model
    # split data into training and validation data, for both features and target
    # The split is based on a random number generator. Supplying a numeric value to
    # the random_state argument guarantees we get the same split every time we
    # run this script.
    # If we do not split the data, we will not be able to evaluate the model correctly.
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
    trained_model = model.fit(train_X, train_y)
    # Predict
    predicted_home_prices = trained_model.predict(val_X)
    
    
    #str_predictions =  ", ".join(map(str, predicted_home_prices.tolist()))

    '''
    We shall evaluate almost every model we build. 
    In most applications, the relevant measure of model quality is 
    predictive accuracy, i.e., how close the predition is to what actually happens.
    
    error=actual−predicted
    With the MAE (Mean Absolute Error) metric, we take the absolute value of each error. 
    This converts each error to a positive number. 
    We then take the average of those absolute errors. 
    This is our measure of model quality. In plain English, it can be said as

    On average, our predictions are off by about X.
    '''
    # Evaluate
    # The error between the predicted and actual home prices is
    mae = mean_absolute_error(val_y, predicted_home_prices)
    response = {
        "predictions": predicted_home_prices.tolist(),
        "mean_absolute_error": mae
    }
    #TODO: https://www.kaggle.com/code/elprofe55or/exercise-model-validation/edit

    return response









