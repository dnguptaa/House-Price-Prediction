import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

df = pd.read_csv("house_project/static/data/data.csv")

# null data in column in percentage
null_var = df.isnull().sum() / df.shape[0] * 100

# Dropping the columns which have more than 20% of null data
drop_columns = null_var[null_var > 20].keys()
df_new = df.drop(columns=drop_columns)

# split the data with object, numeric type column
df_cat_var = df_new.select_dtypes(include=["object"])
df_num_var = df_new.select_dtypes(include=["int64", "float64"])

# get the columns which have null values
missing_cat_vars = [
    var for var in df_cat_var.columns if df_cat_var[var].isnull().sum() > 0
]

missing_num_vars = [
    var for var in df_num_var.columns if df_num_var[var].isnull().sum() > 0
]

# using mean, mode strategy to impute at null places for num and cat values
mode_imputer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])
mean_imputer = Pipeline(steps=[("imputer", SimpleImputer())])

preprocessor = ColumnTransformer(
    transformers=[
        ("mode_imputer", mode_imputer, missing_cat_vars),
        ("mean_imputer", mean_imputer, missing_num_vars),
    ]
)
preprocessor.fit(df_new)

print(
    preprocessor.named_transformers_["mode_imputer"].named_steps["imputer"].statistics_
)
print(
    preprocessor.named_transformers_["mean_imputer"].named_steps["imputer"].statistics_
)

# dataset for the columns whose null value has been imputed
df_clean = preprocessor.transform(df_new)
df_clean_miss_var = pd.DataFrame(df_clean, columns=missing_cat_vars + missing_num_vars)

# updating the dataset
df_new[missing_cat_vars + missing_num_vars] = df_clean

# encoding cat varibales
df_enc = pd.get_dummies(df_new)

# splitting the data in input vars, output
X = df_enc.drop("SalePrice", axis=1)
y = df_enc["SalePrice"]

# splitting the data in training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=51
)

print(f"X_train shape is: {X_train.shape}")
print(f"X_test shape is: {X_test.shape}")
print(f"y_train shaape is: {y_train.shape}")
print(f"y_test shape is: {y_test.shape}")

# linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)


def get_df_columns():
    columns = list(df_enc.columns)

    # removing SalePrice column since it is the result
    columns.remove("SalePrice")
    return columns


def get_test_value():
    return list(X_test.iloc[0, :])


def get_result(test_data):
    pred = lr.predict([test_data])
    pred = round(pred[0], 2)

    accuracy = round(lr.score(X_test, y_test) * 100, 2)
    return pred, accuracy
