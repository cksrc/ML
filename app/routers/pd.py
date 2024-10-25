
import pandas as pd
from fastapi import APIRouter, Depends
from app.dependencies.kaggle import KaggleClient

router = APIRouter(
    prefix="/pd",
    tags=["pd"],
    dependencies=[Depends(KaggleClient)],
    responses={404: {"llm": "Not found"}},
)



@router.get("/")
def entry():
    #Create a pandas dataframe from scratch.
    df = pd.DataFrame({'Column A': ['ColumnA_row_0', 'ColumnA_row_1'], 'Column B': ['ColumnB_row_0', 'ColumnB_row_1']})
    #or 
    df = pd.DataFrame([['ColumnA_row_0', 'ColumnA_row_1'], ['ColumnB_row_0', 'ColumnB_row_1']], columns = ['ColumnA', 'ColumnB'], index=['row 0', 'row 1'])
    columnA_row_0 = df['ColumnA']['row 0']
    print(columnA_row_0)

#    # A series is a column of a DataFrame but it does not have a name
#    # a DataFrame as actually being just a bunch of Series "glued together".
#    new_series = pd.Series(['ColumnC_row_0', 'ColumnC_row_1'])
#    df['Column C'] = new_series

#    #Reading from csv
#    path:str = KaggleClient().dataset_download("dansbecker/melbourne-housing-snapshot")
#    #There are several params when using read_csv, one of the can define an existing column as index: index_col= 
#    csv_df = pd.read_csv(path+"/melb_data.csv") 

#    #We can use the shape attribute to check how large the resulting DataFrame is: (rows, cols)
#    print(csv_df.shape)
#    #Grab the first x amount of rows
#    first_five = csv_df.head()



   
#    response = {
#       "column_a_row_1": column_a_row_1,
#       'column_c_row_0': df['Column C'][0],
#       "table": df
#    }
    return ""