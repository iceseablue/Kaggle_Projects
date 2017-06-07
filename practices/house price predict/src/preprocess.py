# -*- coding: utf-8 -*-
# !/usr/bin/env python
import pandas as pd
import seaborn as sns
import numpy as np
# from sklearn.preprocessing import norm
import scipy
# import stats
import warnings

from rope.refactor import inline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
from scipy.stats import skew
# from scipy.stats.stats import pearsonr

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

import xgboost as xgb


def get_data_from_csv(path):
    data = pd.read_csv(path)
    print type(data)
    print data.keys()



    return data

def analysis_data(data):
    '''
    print data['SalePrice'].describe()
    sns.distplot(data['SalePrice'])

    print("Skewness: %f" % data['SalePrice'].skew())
    print("Kurtosis: %f" % data['SalePrice'].kurt())

    var = 'GrLivArea'
    d = pd.concat([data['SalePrice'], data[var]], axis=1)
    d.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

    var = 'OverallQual'
    data2 = pd.concat([data['SalePrice'], data[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="SalePrice", data=data2)
    fig.axis(ymin=0, ymax=800000)

    var = 'YearBuilt'
    data3 = pd.concat([data['SalePrice'], data[var]], axis=1)
    f, ax = plt.subplots(figsize=(16, 8))
    fig = sns.boxplot(x=var, y="SalePrice", data=data3)
    fig.axis(ymin=0, ymax=800000)
    plt.xticks(rotation=90)


    corrmat = data.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)

    k = 10  # number ofvariables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(data[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                     yticklabels=cols.values, xticklabels=cols.values)


    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(data[cols], size=2.5)



    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data.head(20)
    '''

    # data = pd.get_dummies(data)
    print "HouseStyle:", data['HouseStyle']

    sns.plt.show()


def show_missing(data):
    missing = data.columns[data.isnull().any()].tolist()
    return missing

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, x_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

if __name__ == "__main__":

    df_train = get_data_from_csv('../data/train.csv')
    df_test = get_data_from_csv('../data/test.csv')

    all_data = pd.concat((df_train.loc[:, 'MSSubClass':'SaleCondition'],
                          df_test.loc[:, 'MSSubClass':'SaleCondition']))
    plt.rcParams['figure.figsize'] = (12.0, 6.0)
    prices = pd.DataFrame({"price": df_train["SalePrice"], "log(price + 1)": np.log1p(df_train["SalePrice"])})
    prices.hist()

    # log transform the target:
    df_train["SalePrice"] = np.log1p(df_train["SalePrice"])

    # log transform skewed numeric features:
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

    skewed_feats = df_train[numeric_feats].apply(lambda x: skew(x.dropna()))  # compute skewness
    skewed_feats = skewed_feats[skewed_feats > 0.75]
    skewed_feats = skewed_feats.index

    all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

    all_data = pd.get_dummies(all_data)
    # filling NA's with the mean of the column:
    all_data = all_data.fillna(all_data.mean())

    # creating matrices for sklearn:
    x_train = all_data[:df_train.shape[0]]
    x_test = all_data[df_train.shape[0]:]
    y_train = df_train.SalePrice

    model_ridge = Ridge()
    alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
    cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean()
                for alpha in alphas]

    cv_ridge = pd.Series(cv_ridge, index=alphas)
    cv_ridge.plot(title="Validation - Just Do It")
    plt.xlabel("alpha")
    plt.ylabel("rmse")

    cv_ridge.min()

    model_lasso = LassoCV(alphas=[1, 0.1, 0.001, 0.0005]).fit(x_train, y_train)

    rmse_cv(model_lasso).mean()

    coef = pd.Series(model_lasso.coef_, index=x_train.columns)

    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")

    imp_coef = pd.concat([coef.sort_values().head(10),
                          coef.sort_values().tail(10)])

    plt.rcParams['figure.figsize'] = (8.0, 10.0)
    imp_coef.plot(kind="barh")
    plt.title("Coefficients in the Lasso Model")

    # let's look at the residuals as well:
    plt.rcParams['figure.figsize'] = (6.0, 6.0)

    preds = pd.DataFrame({"preds": model_lasso.predict(x_train), "true": y_train})
    preds["residuals"] = preds["true"] - preds["preds"]
    preds.plot(x="preds", y="residuals", kind="scatter")

    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test)

    params = {"max_depth": 2, "eta": 0.1}
    model = xgb.cv(params, dtrain, num_boost_round=500, early_stopping_rounds=100)

    model.loc[30:, ["test-rmse-mean", "train-rmse-mean"]].plot()
    model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)  # the params were tuned using xgb.cv
    model_xgb.fit(x_train, y_train)
    xgb_preds = np.expm1(model_xgb.predict(x_test))
    lasso_preds = np.expm1(model_lasso.predict(x_test))
    predictions = pd.DataFrame({"xgb": xgb_preds, "lasso": lasso_preds})
    predictions.plot(x="xgb", y="lasso", kind="scatter")
    preds = 0.7 * lasso_preds + 0.3 * xgb_preds

    solution = pd.DataFrame({"id": df_test.Id, "SalePrice": preds})
    solution.to_csv("ridge_sol.csv", index=False)

    # analysis_data(df_train)
