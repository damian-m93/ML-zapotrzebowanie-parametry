import pandas as pd
from sklearn import linear_model, svm, ensemble, tree
import numpy as np


def zapisz_df(df, nazwa_pliku):
    """ Zapisuje daną tablicę do pliku """
    sciezka = 'C:/Users/ca125/Desktop/'
    df.to_excel(sciezka + nazwa_pliku)


def mape(y_true, y_pred):
    """ Wyznacza błąd MAPE """
    return np.mean(abs((y_true - y_pred)/y_true)) * 100


def evaluate(model):
    """ Wyznacza błąd MAPE oraz score modelu """
    print('--------')
    print(model)
    print(f'R2 - train set: {model.score(X_train, y_train)}')
    print(f'R2 - test set: {model.score(X_test, y_test)}')
    print(f'MAPE [%] - train + test set: {mape(y, model.predict(X))}')
    print(f'MAPE [%] - test set: {mape(y_test, model.predict(X_test))}')


# Wczytanie pliku źródłowego (wczytać dane uzupełnione, po obróbce)

dane = pd.read_excel('C:/Users/ca125/Desktop/Dane_wejściowe_przygotowane.xlsx', index_col=0)


# Określenie zbiorów X i y

X = dane.loc[:, 'Rok':]
y = dane['Zapotrzebowanie KSE [MW]']

# Określenie zbioru treningowego oraz testowego

X_train = X.loc[:'2015']
X_test = X.loc['2016':]
y_train = y.loc[:'2015']
y_test = y.loc['2016':]

# Utworzenie modeli do oceny

linreg = linear_model.LinearRegression()
linridge = linear_model.Ridge()
sgdregressor = linear_model.SGDRegressor()
lasso = linear_model.Lasso()
elasticnet = linear_model.ElasticNet()
bayesianridge = linear_model.BayesianRidge()
svr_linear = svm.SVR(kernel='linear')  # Bardzo długie obliczenia, wynik optymalny (MAPE 8%)
svr_rbf = svm.SVR(kernel='rbf')  # Bardzo słaby wynik
randomforest = ensemble.RandomForestRegressor()  # Bardzo dobry wynik, MAPE około 1,7%
gradientboosted = ensemble.GradientBoostingRegressor()  # Bardzo dobry wynik, MAPE około 3%
decisiontree = tree.DecisionTreeRegressor()  # Bardzo dobry wynik, MAPE około 1,4%

modele = [linreg, linridge, sgdregressor, lasso, elasticnet, bayesianridge, randomforest, gradientboosted, decisiontree]

# Ocena modeli

for model in modele:
    model.fit(X_train, y_train)
    evaluate(model)

'''
Na podstawie przeprowadzonej wstępnej oceny poszczególnych modeli wybrano:
RandomForestRegressor / GradientBoostingRegressor
'''



