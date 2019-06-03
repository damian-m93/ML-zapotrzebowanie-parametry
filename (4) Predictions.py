import pandas as pd
from sklearn import ensemble
import numpy as np


def zapisz_df(df, nazwa_pliku):
    """ Zapisuje daną tablicę do pliku """
    sciezka = 'C:/Users/ca125/Desktop/Model_zmiana_czasu/'
    df.to_excel(sciezka + nazwa_pliku)


def mape(y_true, y_pred):
    """ Wyznacza błąd MAPE """
    return np.mean(abs((y_true - y_pred)/y_true)) * 100


# Wczytanie pliku źródłowego (wczytać dane uzupełnione, po obróbce)

dane_uczenie = pd.read_excel('C:/Users/ca125/Desktop/Model_zmiana_czasu/Dane_wejściowe_przygotowane.xlsx', index_col=0)

# Określenie zbiorów X i y - uczenie

X_train = dane_uczenie.loc[:, 'Rok':]
y_train = dane_uczenie['Zapotrzebowanie KSE [MW]']
X_train_3_lata = dane_uczenie[dane_uczenie.index.year >= 2016].loc[:, 'Rok':]
y_train_3_lata = dane_uczenie[dane_uczenie.index.year >= 2016]['Zapotrzebowanie KSE [MW]']
# Wczytanie plików dla zmian czasu

df_test_letni = pd.read_excel('C:/Users/ca125/Desktop/Model_zmiana_czasu/Dane_wejściowe_czas_letni.xlsx', index_col=0)
df_test_zimowy = pd.read_excel('C:/Users/ca125/Desktop/Model_zmiana_czasu/Dane_wejściowe_czas_zimowy.xlsx', index_col=0)

# Określenie zbiorów X (i y?) - predictions

X_test_letni = df_test_letni.loc[:, 'Rok':]
y_test_letni = df_test_letni['Zapotrzebowanie KSE [MW]']
X_test_zimowy = df_test_zimowy.loc[:, 'Rok':]
y_test_zimowy = df_test_zimowy['Zapotrzebowanie KSE [MW]']

# Utworzenie modeli

rfreg_base = ensemble.RandomForestRegressor(n_estimators=10, random_state=42)
# rfreg_final = ensemble.RandomForestRegressor(n_estimators=600,
#                                              criterion='mse',
#                                              max_depth=60,
#                                              min_samples_split=2,
#                                              min_samples_leaf=2,
#                                              min_weight_fraction_leaf=0.0,
#                                              max_features='auto',
#                                              max_leaf_nodes=None,
#                                              min_impurity_decrease=0.0,
#                                              min_impurity_split=None,
#                                              bootstrap=False,
#                                              oob_score=False,
#                                              n_jobs=-1,
#                                              random_state=42,
#                                              verbose=2,
#                                              warm_start=False)
# gradientboosted = ensemble.GradientBoostingRegressor()

rfreg_base.fit(X_train, y_train)
# rfreg_final.fit(X_train, y_train)

# Przygotowanie tabel - LATO

df_pred_lato = pd.DataFrame(rfreg_base.predict(X_test_letni))
df_pred_lato.index = y_train_3_lata.index
roczna_A_predykcje_lato = df_pred_lato.groupby(df_pred_lato.index.year).sum()

roczna_A_real_przes_lato = y_test_letni.groupby(y_test_letni.index.year).sum()

# Przygotowanie tabel - ZIMA

df_pred_zima = pd.DataFrame(rfreg_base.predict(X_test_zimowy))
df_pred_zima.index = y_train_3_lata.index
roczna_A_predykcje_zima = df_pred_zima.groupby(df_pred_zima.index.year).sum()

roczna_A_real_przes_zima = y_test_zimowy.groupby(y_test_zimowy.index.year).sum()

# Przygotowanie tabel - część wspólna

df_pred_on_train_set = pd.DataFrame(rfreg_base.predict(X_train_3_lata))
df_pred_on_train_set.index = y_train_3_lata.index
roczna_A_predykcje_on_train_set = df_pred_on_train_set.groupby(df_pred_on_train_set.index.year).sum()

roczna_A_real = y_train_3_lata.groupby(y_train_3_lata.index.year).sum()

# Utworzenie i zapis sumarycznych tabeli z wynikami - LATO

'''
Model: rfreg_base
Predykcje dla pozostania przy czasie letnim w latach 2016-2018
'''

df_wyniki_1_lato = roczna_A_predykcje_lato.copy().join(roczna_A_real).join(roczna_A_real_przes_lato, rsuffix='_2').join(
    roczna_A_predykcje_on_train_set, rsuffix='_2')
df_wyniki_1_lato.rename({'0': 'Energia w roku - prognoza modelu',
                         'Zapotrzebowanie KSE [MW]': 'Energia w roku - wartości rzeczywiste',
                         'Zapotrzebowanie KSE [MW]_2': 'Energia w roku - wartości rzeczywiste przesunięte o godzinę',
                         '0_2': 'Energia w roku - prognoza modelu na danych uczących'}, axis='columns', inplace=True)
zapisz_df(df_wyniki_1_lato, 'Wyniki_energia_w_roku_wariant_letni.xlsx')

wyniki_mape_lato = []
for y_true in [y_train_3_lata, y_test_letni, rfreg_base.predict(X_train_3_lata)]:
    wyniki_mape_lato.append(mape(y_true, rfreg_base.predict(X_test_letni)))
df_wyniki_2_lato = pd.DataFrame({'PRED / Wartości rzeczywiste': [wyniki_mape_lato[0]],
                                 'PRED / Wartości rzeczywiste przesunięte o godzinę': [wyniki_mape_lato[1]],
                                 'PRED / Prognoza modelu na danych uczących': [wyniki_mape_lato[2]]})
df_wyniki_2_lato.index = ['MAPE [%]']
zapisz_df(df_wyniki_2_lato, 'Wyniki_MAPE_wariant_letni.xlsx')

df_wyniki_3_lato = df_pred_lato.copy().join(y_train_3_lata).join(y_test_letni, rsuffix='_2').join(
    df_pred_on_train_set, rsuffix='_2')
df_wyniki_3_lato.rename({'0': 'Zapotrzebowanie - prognoza modelu',
                         'Zapotrzebowanie KSE [MW]': 'Zapotrzebowanie - wartości rzeczywiste',
                         'Zapotrzebowanie KSE [MW]_2': 'Zapotrzebowanie - wartości rzeczywiste przesunięte o godzinę',
                         '0_2': 'Zapotrzebowanie - prognoza modelu na danych uczących'}, axis='columns', inplace=True)
zapisz_df(df_wyniki_3_lato, 'Wyniki_zapotrzebowanie_KSE_wariant_letni.xlsx')

# Utworzenie i zapis sumarycznych tabel z wynikami - ZIMA

'''
Model: rfreg_base
Predykcje dla pozostania przy czasie zimowym w latach 2016-2018
'''

df_wyniki_1_zima = roczna_A_predykcje_zima.copy().join(roczna_A_real).join(roczna_A_real_przes_zima, rsuffix='_2').join(
    roczna_A_predykcje_on_train_set, rsuffix='_2')
df_wyniki_1_zima.rename({'0': 'Energia w roku - prognoza modelu',
                         'Zapotrzebowanie KSE [MW]': 'Energia w roku - wartości rzeczywiste',
                         'Zapotrzebowanie KSE [MW]_2': 'Energia w roku - wartości rzeczywiste przesunięte o godzinę',
                         '0_2': 'Energia w roku - prognoza modelu na danych uczących'}, axis='columns', inplace=True)
zapisz_df(df_wyniki_1_zima, 'Wyniki_energia_w_roku_wariant_zimowy.xlsx')

wyniki_mape_zima = []
for y_true in [y_train_3_lata, y_test_zimowy, rfreg_base.predict(X_train_3_lata)]:
    wyniki_mape_zima.append(mape(y_true, rfreg_base.predict(X_test_zimowy)))
df_wyniki_2_zima = pd.DataFrame({'PRED / Wartości rzeczywiste': [wyniki_mape_zima[0]],
                                 'PRED / Wartości rzeczywiste przesunięte o godzinę': [wyniki_mape_zima[1]],
                                 'PRED / Prognoza modelu na danych uczących': [wyniki_mape_zima[2]]})
df_wyniki_2_zima.index = ['MAPE [%]']
zapisz_df(df_wyniki_2_zima, 'Wyniki_MAPE_wariant_zimowy.xlsx')

df_wyniki_3_zima = df_pred_zima.copy().join(y_train_3_lata).join(y_test_zimowy, rsuffix='_2').join(
    df_pred_on_train_set, rsuffix='_2')
df_wyniki_3_zima.rename({'0': 'Zapotrzebowanie - prognoza modelu',
                         'Zapotrzebowanie KSE [MW]': 'Zapotrzebowanie - wartości rzeczywiste',
                         'Zapotrzebowanie KSE [MW]_2': 'Zapotrzebowanie - wartości rzeczywiste przesunięte o godzinę',
                         '0_2': 'Zapotrzebowanie - prognoza modelu na danych uczących'}, axis='columns', inplace=True)
zapisz_df(df_wyniki_3_zima, 'Wyniki_zapotrzebowanie_KSE_wariant_zimowy.xlsx')

