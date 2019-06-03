import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def zapisz_df(df, nazwa_pliku):
    """ Zapisuje daną tablicę do pliku """
    sciezka = 'C:/Users/ca125/Desktop/'
    df.to_excel(sciezka + nazwa_pliku)


# Wczytanie pliku źródłowego (wczytać dane wejściowe przed obróbką)

dane = pd.read_excel('C:/Users/ca125/Desktop/Dane_wejściowe.xlsx', index_col=0)

# Uzupełnienie danych wejściowych

dane['Rok'] = dane.index.year
dane['Miesiąc'] = dane.index.month
dane['Dzień w roku'] = dane.index.dayofyear
dane['Dzień w miesiącu'] = dane.index.day
dane['Dzień w tygodniu'] = dane.index.dayofweek
dane['Godzina w dniu'] = dane.index.hour

# Przygotowanie i przekształcenie danych wejściowych

dane.drop(columns=['Dzień'], inplace=True)
dane['Typ dnia [R/W]'] = OrdinalEncoder().fit_transform(dane['Typ dnia [R/W]'].values.reshape(-1, 1))
dane['Typ dnia [R/W]'] = dane['Typ dnia [R/W]'].astype(int)
dane['Święto [S/NS/Wig]'] = OrdinalEncoder().fit_transform(dane['Święto [S/NS/Wig]'].values.reshape(-1, 1))
dane['Święto [S/NS/Wig]'] = dane['Święto [S/NS/Wig]'].astype(int)
column_order = ['Zapotrzebowanie KSE [MW]',
                'Rok',
                'Miesiąc',
                'Dzień w roku',
                'Dzień w miesiącu',
                'Dzień w tygodniu',
                'Godzina w dniu',
                'Godzina w roku',
                'Typ dnia [R/W]',
                'Święto [S/NS/Wig]',
                'Długość dnia',
                'Wschód słońca',
                'Zachód słońca',
                'Temperatura powietrza [°C]',
                'Wysokość podstawy chmur CL CM szyfrowana [kod]',
                'Widzialność [kod]',
                'Kierunek wiatru [°]',
                'Prędkość wiatru [m/s]',
                'Poryw wiatru [m/s]',
                'Temperatura termometru zwilżonego [°C]',
                'Ciśnienie pary wodnej [hPa]',
                'Wilgotność względna [%]',
                'Temperatura punktu rosy [°C]',
                'Ciśnienie na pozimie stacji [hPa]',
                'Ciśnienie na pozimie morza [hPav',
                'Pogoda bieżąca [kod]',
                'Pogoda ubiegła [kod]',
                'Stopniodni grzewcze [18 °C Baza]',
                'Stopniodni chłodnicze [18 °C Baza]']
dane = dane[column_order]

# Zapis pliku

zapisz_df(dane, 'Dane_wejściowe_2.xlsx')