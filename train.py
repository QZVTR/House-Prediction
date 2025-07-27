import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing   
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import data_setup



def splitData(df):
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    y_log = np.log1p(y)  # Log-transform target
    Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(X, y_log, test_size=0.2, random_state=42)
    return Xtrain, Xtest, ytrain, ytest

def scaleData(Xtrain, Xtest):
    scaler = preprocessing.StandardScaler()
    XTrainScaled = scaler.fit_transform(Xtrain)
    XTestScaled = scaler.transform(Xtest)
    return XTrainScaled, XTestScaled, scaler

def evaluateModel(model, Xtest, ytest, modelName='Model', is_pytorch_model=False):
    # Convert Xtest to tensor for PyTorch
    if is_pytorch_model:
        Xtest_tensor = torch.tensor(Xtest, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            ypred_log = model(Xtest_tensor).numpy().flatten()
    else:
        ypred_log = model.predict(Xtest)
    # Inverse-transform predictions and test set
    ypred = np.expm1(ypred_log)
    ytest_orig = np.expm1(ytest)
    mae = mean_absolute_error(ytest_orig, ypred)
    mse = mean_squared_error(ytest_orig, ypred)
    rmse = mse ** 0.5
    print(f'\n{modelName} Evaluation Metrics:')
    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    return ypred

def printClassificationReport(ytest, ypred, modelName='Model'):
    print(f"\nClassification Report for {modelName}:")
    print(classification_report(ytest, ypred, zero_division=0))
    cm = confusion_matrix(ytest, ypred)
    print(f"Confusion Matrix for {modelName}:")
    print(cm)

def saveModel(model, filename, is_pytorch_model=False):
    if is_pytorch_model:
        torch.save(model.state_dict(), filename)
        print(f'PyTorch model saved to {filename}')
    else:
        joblib.dump(model, filename)
        print(f'Model saved to {filename}')

def loadModel(filename, model_class=None, is_pytorch_model=False, input_dim=None):
    if is_pytorch_model:
        model = model_class(input_dim)
        model.load_state_dict(torch.load(filename))
        model.eval()
        print(f'PyTorch model loaded from {filename}')
        return model
    else:
        model = joblib.load(filename)
        print(f'Model loaded from {filename}')
        return model

def printFeatureImportance(model, Xtrain, modelName):
    importances = model.feature_importances_
    feature_names = Xtrain.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    print(f"\nFeature Importance for {modelName}:")
    print(importance_df.sort_values(by='Importance', ascending=False))

def tuneModel(model, param_grid, Xtrain, ytrain):
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)
    print(f"Best parameters for {model.__class__.__name__}: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Define PyTorch Neural Network
class HousingNN(nn.Module):
    def __init__(self, input_dim):
        super(HousingNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.output(x)
        return x

def trainNN(model, Xtrain, ytrain, Xval, yval, epochs=100, batch_size=32, patience=10):
    # Convert to tensors
    Xtrain_tensor = torch.tensor(Xtrain, dtype=torch.float32)
    ytrain_tensor = torch.tensor(ytrain.values, dtype=torch.float32).reshape(-1, 1)
    Xval_tensor = torch.tensor(Xval, dtype=torch.float32)
    yval_tensor = torch.tensor(yval.values, dtype=torch.float32).reshape(-1, 1)
    
    # Create data loaders
    train_dataset = TensorDataset(Xtrain_tensor, ytrain_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(Xval_tensor)
            val_loss = criterion(val_outputs, yval_tensor).item()

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def main():
    # Load data
    dataDf = data_setup.loadData()
    #print("Dataset Preview:")
    #print(dataDf.head())
    
    if dataDf.isnull().any().any():
        print("Warning: Missing values detected after preprocessing.")

    # Split data
    Xtrain, Xtest, ytrain, ytest = splitData(dataDf)
    # Create validation set for NN
    #XtrainNN, XvalNN, ytrainNN, yvalNN = model_selection.train_test_split(Xtrain, ytrain, test_size=0.1, random_state=42)
    XTrainScaled, XTestScaled, scaler = scaleData(Xtrain, Xtest)
    #XValScaled = scaler.transform(XvalNN)

    
    # RandomForestRegressor
    """rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rfModel = tuneModel(RandomForestRegressor(random_state=42), rf_param_grid, XTrainScaled, ytrain)
    rfYPred = evaluateModel(rfModel, XTestScaled, ytest, "RandomForestRegressor")
    printFeatureImportance(rfModel, Xtrain, "RandomForestRegressor")
    saveModel(rfModel, 'rf_model.pkl')"""

    # XGBRegressor
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.3]
    }
    xgbModel = tuneModel(XGBRegressor(random_state=42), xgb_param_grid, XTrainScaled, ytrain)
    xgbYPred = evaluateModel(xgbModel, XTestScaled, ytest, "XGBRegressor")
    printFeatureImportance(xgbModel, Xtrain, "XGBRegressor")
    #saveModel(xgbModel, 'xgb_model_new.pkl')
    

    """
    # Neural Network (PyTorch)
    input_dim = XTrainScaled.shape[1]
    nnModel = HousingNN(input_dim)
    nnModel = trainNN(nnModel, XTrainScaled, ytrain, XValScaled, yvalNN, epochs=100, batch_size=32, patience=10)
    nnYPred = evaluateModel(nnModel, XTestScaled, ytest, "NeuralNetwork", is_pytorch_model=True)
    saveModel(nnModel, 'nn_model.pth', is_pytorch_model=True)
    
    """

    #saveModel(scaler, 'scaler_new.pkl')

if __name__ == "__main__":
    main()