from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

ride=pd.read_csv('ride.csv')
ride=ride[ride['price']!=0]
enc1 = LabelEncoder()
ride['source_id'] = enc1.fit_transform(ride['source'])
np.save('class_source.npy', enc1.classes_)
enc2 = LabelEncoder()
ride['destination_id'] = enc2.fit_transform(ride['destination'])
np.save('class_destination.npy', enc2.classes_)
enc3 = LabelEncoder()
ride['cab_type_id'] = enc3.fit_transform(ride['cab_type'])
np.save('class_cab.npy', enc3.classes_)
ride=ride.drop(['source','destination','cab_type'],axis=1)


print(ride.head())
labels = np.array(ride['price'])

features= ride.drop(['price'], axis = 1)

feature_list = list(features.columns)
print(features.columns)
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)
print(predictions)
joblib.dump(rf, "ride_price.joblib")