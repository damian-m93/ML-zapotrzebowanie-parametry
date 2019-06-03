import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


def zapisz_df(df, nazwa_pliku):
    """ Zapisuje daną tablicę do pliku """
    sciezka = 'C:/Users/ca125/Desktop/'
    df.to_excel(sciezka + nazwa_pliku)


def mape(y_true, y_pred):
    """ Wyznacza błąd MAPE """
    return np.mean(abs((y_true - y_pred)/y_true)) * 100


def evaluate1(model):
    """ Wyznacza błąd MAPE oraz score modelu """
    print('--------')
    print(model)
    print(f'R2 - train set: {model.score(X_train, y_train)}')
    print(f'R2 - test set: {model.score(X_test, y_test)}')
    print(f'MAPE [%] - train + test set: {mape(y, model.predict(X))}')
    print(f'MAPE [%] - test set: {mape(y_test, model.predict(X_test))}')


def evaluate2(model, X_test, y_test):
    predictions = model.predict(X_test)
    wynik_mape = mape(y_test, predictions)
    print(f'MAPE [%]: {wynik_mape}')
    return wynik_mape


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

# Utworzenie modeli

rfreg = ensemble.RandomForestRegressor()
gradientboosted = ensemble.GradientBoostingRegressor()

# Cross Validation -

dane_idx = dane.copy()
dane_idx['Data'] = dane_idx.index
dane_idx.index = pd.Series(dane_idx.index).apply(lambda row: dane_idx.index.get_loc(row))
cv_sets = []
lista_lat = list(dane.index.year.unique())
for rok in lista_lat[5:-1]:
    train_idx = dane_idx[dane_idx['Data'].dt.year <= rok].index.values.astype(int)
    test_idx = dane_idx[dane_idx['Data'].dt.year == rok+1].index.values.astype(int)
    cv_sets.append((train_idx, test_idx))

# Random Search

# Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
n_estimators = [200, 600, 800]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
# max_depth.append(None)
max_depth = [10, 30, 50, None]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rfreg_random = RandomizedSearchCV(estimator=rfreg,
                                  param_distributions=random_grid,
                                  n_iter=3,
                                  cv=cv_sets,
                                  verbose=2,
                                  random_state=42,
                                  n_jobs=-1)

rfreg_random.fit(X, y)
print('Najlepsze parametry:')
print(rfreg_random.best_estimator_)

base_model = ensemble.RandomForestRegressor(n_estimators=10, random_state=42)
base_model.fit(X_train, y_train)
base_accuracy = evaluate2(base_model, X_test, y_test)
rfreg_final = ensemble.RandomForestRegressor(n_estimators=600,
                                             criterion='mse',
                                             max_depth=60,
                                             min_samples_split=2,
                                             min_samples_leaf=2,
                                             min_weight_fraction_leaf=0.0,
                                             max_features='auto',
                                             max_leaf_nodes=None,
                                             min_impurity_decrease=0.0,
                                             min_impurity_split=None,
                                             bootstrap=False,
                                             oob_score=False,
                                             n_jobs=-1,
                                             random_state=42,
                                             verbose=2,
                                             warm_start=False)

rfreg_final.fit(X_train, y_train)
rfreg_final_accuracy = evaluate2(rfreg_final, X_test, y_test)
