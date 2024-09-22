import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import statistics
import pickle


class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self, delimiter):
        self.data = pd.read_csv(self.file_path,delimiter=delimiter)
        
    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)
    


class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.createModel()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5

    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)
        
    def createModeFromColumn(self,col):
        return statistics.mode(self.x_train[col])

    def removeColumn(self,col):
        self.x_train.drop(col, axis=1, inplace=True)
        self.x_train.drop(col, axis=1, inplace=True)

    def createMeanFromColumn(self,col):
        return np.mean(self.x_train[col])

    def fillingNA(self,col,value):
        self.x_train[col].fillna(value, inplace=True)
        self.x_test[col].fillna(value, inplace=True)

    def encodeBinary(self,col):
        self.train_encode={"Geography": {"Germany":0,"France":1,"Spain":2}, "Gender": {"Male":0,"Female":1}}
        self.test_encode={"Geography": {"Germany":0,"France":1,"Spain":2}, "Gender": {"Male":0,"Female":1}}
        self.x_train=self.x_train.replace(self.train_encode)
        self.x_test=self.x_test.replace(self.test_encode)

    def selectDesiredFeatures(self, feature_list):
        self.x_train = self.x_train[feature_list]
        self.x_test = self.x_test[feature_list]

    def createModel(self):
        self.model = xgb.XGBClassifier(n_estimators=4, max_depth=6, learning_rate=0.5, objective="binary:logistic", random_state=42)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions)
    
    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test) 
    
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict, target_names=['A','B']))

    def save_model_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

file_path = 'data_C.csv'
data_handler = DataHandler(file_path)
data_handler.load_data(delimiter=',')
data_handler.create_input_output('churn')
input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)
model_handler.split_data()

#Handling missing values
creditscore_replace_NA = model_handler.createMeanFromColumn('CreditScore')
model_handler.fillingNA('CreditScore',creditscore_replace_NA)

#Feature encoding
model_handler.encodeBinary(['Geography', 'Gender'])

#Feature selection
features_list = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance','NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

model_handler.selectDesiredFeatures(features_list)

#Prediction model
model_handler.train_model()
model_handler.makePrediction()
model_handler.createReport()

#Save model to pickle
model_handler.save_model_to_file('2602148814.pkl')