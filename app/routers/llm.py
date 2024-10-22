from fastapi import APIRouter, Depends
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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
    

    #Model Fine Tuning
    # Overfitting: is the situation where a model matches the training data almost perfectly, 
    # but does poorly in validation and other new data. 
    # Underfitting: When a model fails to capture important distinctions and patterns in the data, 
    # so it performs poorly even in training data
    # Find the best tree size among possible options
    possible_leaf_nodes = [5, 50, 5000, 5000]
    leaf_nodes_to_mae = {}
    for leaf_nodes in possible_leaf_nodes:
        leaf_nodes_to_mae[leaf_nodes] = get_mae(leaf_nodes, train_X=train_X, val_y=val_y, train_y=train_y, val_X=val_X)
    
    best_tree_size = min(leaf_nodes_to_mae, key=leaf_nodes_to_mae.get)
    
    

    # The random forest uses many trees, 
    # and it makes a prediction by averaging the predictions of each component tree. 
    # It generally has much better predictive accuracy than a single decision tree 
    # and it works well with default parameters.
    # There are parameters which allow you to change the performance of the Random Forest
    # much as we changed the maximum depth of the single decision tree. 
    # But one of the best features of Random Forest models is that they generally work 
    # reasonably even without this tuning.
    forest_model =  RandomForestRegressor(random_state=1)
    forest_model.fit(X=train_X, y=train_y)
    random_fores_preds = forest_model.predict(X=val_X)
    random_forest_mae = mean_absolute_error(y_true=val_y, y_pred=random_fores_preds)

    response = {
        "predictions": predicted_home_prices[0:5].tolist(),
        "not_optimized_mean_absolute_error": mae,
        "optimized_mean_absolute_error": leaf_nodes_to_mae[best_tree_size],
        "best_tree_size": best_tree_size,
        "random_forest_mae": random_forest_mae

    }
    return response


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)








