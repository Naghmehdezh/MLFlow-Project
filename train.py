
import os
import pandas as pd
from fbprophet import Prophet
import mlflow
import mlflow.sklearn


data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aligned_dataset.csv")
train_df = pd.read_csv(data_path)

train_df['Date'] = pd.to_datetime(train_df['Date'],format="%m/%d/%Y").dt.date



def item_sales (df,target_attribute,sales_criteria,ranked_items,i):
    item_name=ranked_items[i-1]
    print(item_name)
    df =df[df[target_attribute] == item_name]
    #display(df)
    df=df.groupby([target_attribute,'Date'],as_index=False)[sales_criteria].sum()
    
    df=df.sort_values(['Date'])
    df = df[(df[sales_criteria] >= 0) & (df[sales_criteria] <= 10000)] # remoing the outliers
    return df


# a function for building a forecasting model for 1 item
def build_FBProphet_model(df):
    from fbprophet import Prophet 
    model = Prophet(yearly_seasonality=20, weekly_seasonality=False) 
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5) # these parameters also can be the function input
    model.fit(df)
    return model


# a function for predicting future sales
def predict_future_data(FBProphet_Model,p,f):  
    future_data = FBProphet_Model.make_future_dataframe(periods=p, freq=f)
    sales_df=item_sales(new_tr_df,target_attribute,sales_criteria,sku_list,i)
    forecast_data = FBProphet_Model.predict(future_data)
    FBProphet_Model.plot(forecast_data)
    forecast_data.set_index('ds').yhat.plot()
    FBProphet_Model.plot_components(forecast_data)
    return forecast_data


# building model for historical dataset

sku_list=[22630, 23166, 85099, 84077, 22197, 85123, 84879, 21212, 23084, 22616, 22492, 21977, 22178, 17003, 15036, 22386, 23203, 21915, 20725, 47566]
k=1
target_attribute='StockCode'
sales_criteria='Quantity'
period=90
freq='d'
FBP_DATETIME='ds'
FBP_TARGET='y'
fig_list=[]


if __name__ == "__main__":
	with mlflow.start_run():
	# it automatically terminates the run at the end of the with block
	  for i in range(1,k):
		sales_per_date=item_sales(train_df,target_attribute,sales_criteria,sku_list,i)
		sales_per_date.rename(columns={'Date': FBP_DATETIME, 'Quantity': FBP_TARGET}, inplace=True)
		FBProphet_model = Prophet(yearly_seasonality=True)
		FBProphet_model.fit(sales_per_date)
		sku_name=str(sku_list[i-1])
		mlflow.log_param("yearly_seasonality" , True)
		model_path="/dbfs/mlflow/UK_dataset/uk_data_original1/model-%s-%f" % (sku_name , 1)
		mlflow.sklearn.save_model(FBProphet_model, model_path)
		mlflow.log_artifact(model_path,artifact_path = 'model')
		mlflow.sklearn.log_model(FBProphet_model,artifact_path = 'model')

 

