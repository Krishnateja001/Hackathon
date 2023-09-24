import numpy as np 
import pandas as pd
from datetime import datetime
on_board=pd.read_csv("MBTA_Commuter_Rail_Ridership_by_Service_Date_and_Line.csv")

travel_time=pd.read_csv("MBTA_Commuter_Rail_Ridership_by_Trip%2C_Season%2C_Route_Line%2C_and_Stop.csv")

print(on_board.columns)


date_format = '%Y/%m/%d %H:%M:%S%z'
on_board['year']=pd.DatetimeIndex(on_board['service_date']).year
on_board['month']=pd.DatetimeIndex(on_board['service_date']).month
on_board['day']=pd.DatetimeIndex(on_board['service_date']).day
on_board['hour']=pd.DatetimeIndex(on_board['service_date']).hour

on_board=on_board.drop(['service_date'],axis=1)
on_board.to_csv('commuter_count.csv',index=False)