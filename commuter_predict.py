import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
features = pd.read_csv('commuter_count.csv')

features = pd.get_dummies(features)
labels = np.array(features['estimated_boardings'])

features= features.drop(['estimated_boardings','ObjectId'], axis = 1)

feature_list = list(features.columns)
print(features.columns)
features = np.array(features)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
rf.fit(train_features, train_labels)
predictions = rf.predict(test_features)
print(predictions)
# joblib.dump(rf, "commuter_count.joblib",compress= 9)

with open('commuter_count.pkl','wb') as f:
    pickle.dump(rf,f)