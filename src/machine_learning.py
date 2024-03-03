from sklearn import preprocessing
import seaborn as sns
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor, XGBRFRegressor
from sklearn.tree import ExtraTreeRegressor, plot_tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.svm import SVR
import os
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from math import sqrt
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import joblib
from sklearn.metrics import mean_absolute_percentage_error

def preparation(df):
    df = df.drop('reference', axis=1)
    df = df.drop(df[df['type'] == 'Penthouse'].index)
    df = df.drop(df[df['type'] == 'Terraced'].index)
    df = df.drop(df[df['type'] == 'Duplex'].index)
    df = df.drop(df[df['type'] == 'House'].index)
    df = df.drop(df[df['type'] == 'Semi-detached'].index)

    oneonehotencoder = preprocessing.OneHotEncoder()
    threshold = 7
    districts_to_update = df['neighbourhood'].value_counts().loc[lambda x: x <= threshold].index
    df.loc[df['neighbourhood'].isin(districts_to_update), 'neighbourhood'] = 'OTHER'
    df.dropna(subset=['neighbourhood'], inplace=True)
    df.dropna(subset=['plant'], inplace=True)   
    df = df.drop('district', axis=1)

    df_dummy = pd.get_dummies(df)
    df_dummy = df_dummy.drop("lift_no lift", axis=1)
    df_dummy = df_dummy.drop('parking_no', axis=1)
    df_dummy = df_dummy.drop('type_Flat', axis=1)
    df_dummy = df_dummy.drop(df_dummy[df_dummy['price'] >1000000].index)
    return df_dummy


def visuals_1(df):
    sns.heatmap(df.corr())


def visuals_2(df_dummy):
    sns.histplot(x="price", data=df_dummy)
    return df_dummy

def linear_model(df_dummy):
    x,y=df_dummy.drop('price', axis=1), df_dummy.price
    lr_m=Ridge()
    lr_m.fit(x,y)
    y_pred=lr_m.predict(x)
    plt.scatter(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    print("Mean Absolute Error:", mae)


def impact_variables(df_dummy):
    x,y=df_dummy.drop('price', axis=1), df_dummy.price
    vars_pisos=list(df_dummy.columns)
    vars_pisos.remove('price')
    # Create a Ridge regression model
    lr_m = Ridge()
    lr_m.fit(x, y)

    # Get the coefficients (effect) of each variable
    coefficients = lr_m.coef_

    # Create an array of indices for the variables
    indices = np.arange(len(vars_pisos))

    # Sort the variables and coefficients in descending order
    sorted_indices = np.argsort(coefficients)[::-1]
    sorted_vars_pisos = [vars_pisos[i] for i in sorted_indices]
    sorted_coefficients = coefficients[sorted_indices]

    # Plot the effect of each variable
    plt.figure(figsize=(12, 6))
    plt.bar(indices, sorted_coefficients)
    plt.xticks(indices, sorted_vars_pisos, rotation='vertical')
    plt.xlabel('Variable')
    plt.ylabel('Effect on Price')
    plt.title('Effect of Variables on Price')
    plt.tight_layout()
    plt.show()

def linear_model_2(df_dummy):
    X = df_dummy.drop('price', axis=1)
    y = df_dummy.price
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
    reg = LinearRegression().fit(X_train, y_train)
    prediction = reg.predict(X_test)
    score=r2_score(y_test, prediction)
    mae = mean_absolute_error(y_test, prediction)


    print("Mean Absolute Error:", mae)
    print("r2_score", score)
    return df_dummy

def all_models(df_dummy):
    rf = RandomForestRegressor()
    rfc=RandomForestClassifier()
    xgb = XGBRegressor()
    xgbr = XGBRFRegressor()
    linreg = LinearRegression()
    trees = ExtraTreeRegressor()
    knn = KNeighborsRegressor()
    gb = GradientBoostingRegressor()
    regressor = SVR(kernel = 'rbf')
    models = [rf, xgb, xgbr, linreg, trees, knn, gb, regressor]

    models = [rf, xgb, xgbr, linreg, trees, knn, gb,regressor]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

    model_parameter_grid = [
        # Parameter grid for RandomForestRegressor
        {'n_estimators': [100, 200], 'max_depth': [5,30]},
        # Parameter grid for XGBRegressor
        {'n_estimators': [100, 200], 'max_depth': [5,30]},
        # Parameter grid for XGBRFRegressor
        {'n_estimators': [100, 200], 'max_depth': [5,30]},
        # Parameter grid for LinearRegression
        {},
        # Parameter grid for ExtraTreeRegressor
        {'max_depth': [3, 5]},
        # Parameter grid for KNeighborsRegressor
        {'n_neighbors': [3, 5]},
        # Parameter grid for GradientBoostingRegressor
        {'n_estimators': [100, 200], 'max_depth': [5, 30]},
    
        # Parameter grid for SVR
        {'C': [1.0, 2.0], 'gamma': ['scale', 'auto']}]

    for model, param_grid in zip(models, model_parameter_grid):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        y_train_pred = best_model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        y_test_pred = best_model.predict(X_test)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print(f"Model: {type(model).__name__}")
        print("Best Parameters:", grid_search.best_params_)
        print("Train MAE:", train_mae)
        print("Test MAE:", test_mae)
        print()

        return df_dummy
    

def reproduce_model(df_dummy):
    # Load the data and split it into features (X) and target variable (y)
    data = pd.read_csv("../data/dummy.csv")  # Replace 'your_data.csv' with the actual file path or dataset
    X = data.drop('price', axis=1)  # Assuming 'price' is the target variable
    y = data['price']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Adjust test_size as needed

    # Define the base models
    rf_model = RandomForestRegressor()  # You can specify the desired hyperparameters
    xgb_model = XGBRegressor()
    gbm_model = GradientBoostingRegressor()

    # Define the meta-learner model
    meta_model = LinearRegression()

    # Define the stacking ensemble
    ensemble = StackingRegressor(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model),
            ('gbm', gbm_model)
        ],
        final_estimator=meta_model
    )

    # Train the stacking ensemble
    ensemble.fit(X_train, y_train)
    predictions = ensemble.predict(X_test)
    joblib.dump(ensemble, '../trained_model/model_2.pkl')

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print("R2 Score:", r2)
    print("MAE:", mae)
    print("mse:", mse)
    print("MAPE:", mape)
    print("rmse:", rmse)
    
    return ensemble



def last_try(df_dummy):
    data = pd.read_csv("../data/dummy.csv")
    X = data.drop('price', axis=1)  # Assuming 'price' is the target variable
    y = data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100],  # Number of trees
        'max_depth': [None],  # Maximum depth of each tree
        'min_samples_split': [2]  # Minimum number of samples required to split a node
    }

    # Create the RandomForestRegressor
    rf_model = RandomForestRegressor()

    # Perform grid search
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid_search.fit(X, y)

    # Get the best model and best parameter combination
    best_rf_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Print the best parameter combination
    y_pred = best_rf_model.predict(X_test)
    
    joblib.dump(rf_model, '../trained_model/model_2.pkl')

    # Calculate the MAE
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print("R2 Score:", r2)
    print("MAE:", mae)
    print("mse:", mse)
    print("MAPE:", mape)
    print("rmse:", rmse)
    return rf_model





    





