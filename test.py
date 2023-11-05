import pandas as pd
import datetime

csv_datei = 'Realisierte_Erzeugung_202001010000_202212312359_Viertelstunde.csv'

df = pd.read_csv(csv_datei, delimiter=";")

# casting the data to the correct format


df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y')
df['Anfang'] = pd.to_datetime(df['Anfang'], format='%H:%M')
df['Ende'] = pd.to_datetime(df['Ende'], format='%H:%M')

# df['Wasserkraft [MWh] Originalauflösungen'] = df['Wasserkraft [MWh] Originalauflösungen'].str.replace(',', '.').astype(float)

# printing the day of the week
# print(df['Datum'].dt.day_name())

# data for 2020 (via filtering)
# filt_20 = ((df['Datum'] >= pd.to_datetime('01.01.2020')) & (df['Datum'] < pd.to_datetime('01.01.2021')))
# print(df.loc[filt_20])

# setting the date as an index 


df.set_index('Datum', inplace=True)


df['Wasserkraft [MWh] Originalauflösungen'] = df['Wasserkraft [MWh] Originalauflösungen'].str.replace(',', '').astype(float)
# print(water_mean)


print(df['2020-01-01':'2020-12-31']['Wasserkraft [MWh] Originalauflösungen'].sum())

