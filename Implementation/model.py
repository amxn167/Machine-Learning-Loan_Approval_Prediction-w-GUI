from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd

# load the data
data = pd.read_csv('C:/Users/mailm/Downloads/loan_predicition-master/loan_predicition-master/LoanApprovalPrediction.csv')
# Drop Loan_ID column
data.drop(['Loan_ID'], axis=1, inplace=True)
# convert to int datatype
label_encoder = LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
    data[col] = label_encoder.fit_transform(data[col])

# fill in missing rows
for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())
# divide model into features and target variable
x = data.drop(['Loan_Status'], axis=1)
y = data.Loan_Status

# divide into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)
# define the model
modelrfc = RandomForestClassifier()
# fit the model on the training data
modelrfc.fit(x_train, y_train)
#save the train model
with open('train_modelrfc.pkl', mode='wb') as pkl:
    pickle.dump(modelrfc, pkl)


# #Decision Tree Classifier
# modeldtc=DecisionTreeClassifier()
# modeldtc.fit(x_train, y_train)

# #save the train model
# with open('train_modeldtc.pkl', mode='wb') as pkl:
#     pickle.dump(modeldtc, pkl)

#Support Vector Machine Classifier
modeldtc=SVC()
modeldtc.fit(x_train, y_train)

#save the train model
with open('train_modeldtc.pkl', mode='wb') as pkl:
    pickle.dump(modeldtc, pkl)

