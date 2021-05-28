import pandas 
import numpy

ds=pandas.read_csv("50_Startups.csv")
state = ds['State']  
y = ds["Profit"]

from sklearn.preprocessing import LabelEncoder # importing the label encoder for data preprocessing

state_le= LabelEncoder()
state_le_final = state_le.fit_transform(state) # First step is to do label encoding in the 'State Feauture'. Assign a new variable name to it

from sklearn.preprocessing import OneHotEncoder

state_ohe = OneHotEncoder()
state_le_final.reshape(-1,1)    # Convert the 'State' feature into 2D array to do onehotencoding
state_le_final_2D = state_le_final.reshape(-1 ,1)

state_dump = state_ohe.fit_transform(state_le_final_2D)  # assign a variable to dump it to avoid dummy variable trap
state_final = state_dump.toarray()
state_final_trap = state_final[: , 0:2]

X = ds[['R&D Spend', 'Administration', 'Marketing Spend',]]
X_Final = numpy.hstack(( X , state_final_trap))

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_Final , y )
a = model.coef_

print(a)
