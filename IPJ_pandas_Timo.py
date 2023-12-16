import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots 
import streamlit as st
from energyConsumption import energyConsumption

# Interaktiver Benutzereingabe für das Datum
selected_date_str = input("Bitte geben Sie das Datum im Format TT.MM.JJJJ ein: ")
selected_date = datetime.strptime(selected_date_str, "%d.%m.%Y")

start_time = time.time()                      #Startzeit des Programms

# Dateinamen
file_production = 'Realisierte_Erzeugung_202001010000_202212312359_Viertelstunde.csv'
file_consumption = 'Realisierter_Stromverbrauch_202001010000_202212312359_Viertelstunde.csv'

# Einlesen der Daten aus CSV-Dateien
production_df = pd.read_csv(file_production, delimiter=';')
consumption_df = pd.read_csv(file_consumption, delimiter=';')

# Spaltenbezeichnungen
DATE = 'Datum'
STARTTIME = 'Anfang'
BIOMAS = 'Biomasse [MWh] Originalauflösungen'
HYDROELECTRIC = 'Wasserkraft [MWh] Originalauflösungen'
WIND_OFFSHORE = 'Wind Offshore [MWh] Originalauflösungen'
WIND_ONSHORE = 'Wind Onshore [MWh] Originalauflösungen'
PHOTOVOLTAIC = 'Photovoltaik [MWh] Originalauflösungen'
OTHER_RENEWABLE = 'Sonstige Erneuerbare [MWh] Originalauflösungen'
CONSUMPTION = 'Gesamt (Netzlast) [MWh] Originalauflösungen'

# Umwandlung von Datumsspalten in DateTime-Objekte
production_df[DATE] = pd.to_datetime(production_df[DATE], format='%d.%m.%Y')
production_df[STARTTIME] = pd.to_datetime(production_df[STARTTIME], format='%H:%M')
consumption_df[DATE] = pd.to_datetime(consumption_df[DATE], format='%d.%m.%Y')
consumption_df[STARTTIME] = pd.to_datetime(consumption_df[STARTTIME], format='%H:%M')

# Bereinigung von Datenformaten der erneubaren Energien
columns_to_clean = [HYDROELECTRIC, BIOMAS, WIND_OFFSHORE, WIND_ONSHORE, PHOTOVOLTAIC, OTHER_RENEWABLE]
for column in columns_to_clean:
    production_df[column] = production_df[column].str.replace(".", "").str.replace(",", ".").replace('-', 0).astype(float)

# Bereinigung von Datenformaten des Gesamtenstromverbrauches
consumption_df[CONSUMPTION] = consumption_df[CONSUMPTION].str.replace(".", "").str.replace(",", ".").astype(float)

production_df['Total Production'] = production_df[columns_to_clean].sum(axis=1)
production_by_year = production_df.groupby(production_df[DATE].dt.year)['Total Production'].sum()
consumption_by_year = consumption_df.groupby(consumption_df[DATE].dt.year)[CONSUMPTION].sum()


production_by_type_and_year = production_df.groupby(production_df[DATE].dt.year)[columns_to_clean].sum()

pd.options.display.float_format = '{:.2f}'.format  # Set Pandas to display floating-point numbers with two decimal places

data_by_year = {}                                  # Aggregation der Daten nach Jahren und Speicherung in einem Dictionary

for year, data in production_df.groupby(production_df[DATE].dt.year):
    production_data = data[columns_to_clean].sum()
    consumption_data = consumption_df[consumption_df[DATE].dt.year == year][CONSUMPTION]
    total_consumption = consumption_data.sum()
    data_by_year[year] = {'Production': production_data.sum(), 'Consumption': total_consumption, BIOMAS: production_data[BIOMAS], HYDROELECTRIC: production_data[HYDROELECTRIC], WIND_OFFSHORE: production_data[WIND_OFFSHORE], WIND_ONSHORE: production_data[WIND_ONSHORE], PHOTOVOLTAIC: production_data[PHOTOVOLTAIC], OTHER_RENEWABLE: production_data[OTHER_RENEWABLE]}


for year, data in data_by_year.items():             # Ausgabe der aggregierten Daten pro Jahr
    print(f"Year: {year}")
    print(f"Total Renewable Energy Production: {data['Production']} MWh")
    print(f"Total Consumption: {data['Consumption']} MWh")
    print(f"Biomasse: {data[BIOMAS]} MWh")
    print(f"Wasserkraft: {data[HYDROELECTRIC]} MWh")
    print(f"Wind Offshore: {data[WIND_OFFSHORE]} MWh")
    print(f"Wind Onshore: {data[WIND_ONSHORE]} MWh")
    print(f"Photovoltaik: {data[PHOTOVOLTAIC]} MWh")
    print(f"Sonstige Erneuerbare: {data[OTHER_RENEWABLE]} MWh")
    print()

total_renewable_production = production_df[columns_to_clean].sum(axis=1)
total_consumption = consumption_df[CONSUMPTION]

# Filtern der Daten für das ausgewählte Datum
selected_production = production_df[production_df[DATE] == selected_date]
selected_consumption = consumption_df[consumption_df[DATE] == selected_date]

total_renewable_production_selected_date = selected_production[columns_to_clean].sum(axis=1).sum()
print(f"Summe der erneuerbaren Energien am {selected_date_str}: {total_renewable_production_selected_date} MWh")


end_time = time.time()                         # The time at the end of the program is stored
duration = end_time - start_time               # Duration of the program is calculated
print("Duration of the program: ", round(duration, 2))

# Berechnung der prozentualen Anteile der erneuerbaren Energieerzeugung am Gesamtverbrauch
percent_renewable = total_renewable_production / total_consumption * 100 

counts, intervals = np.histogram(percent_renewable, bins = np.arange(0, 111, 1))  # Use NumPy to calculate the histogram of the percentage distribution

x = intervals[:-1]                               # Define the x-axis values as the bin edges
labels = [f'{i}%' for i in range(0, 111, 1)]     # Create labels for x-axis ticks (von 0 bis 111 in Einzelnschritten)

fig = go.Figure(data=[go.Bar(x=x, y=counts)])    # Create a bar chart using Plotly

fig.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(0, 111, 5)), ticktext=labels[::5]))  # X-axis label settings

# Title and axis labels settings
fig.update_layout(title='Anzahl der Viertelstunden in Jahren 2020-2022 mit 0-110 % EE-Anteilen',
                  xaxis_title='Prozentsatz erneuerbarer Energie',
                  yaxis_title='Anzahl der Viertelstunden')

fig.show()
 
# Plotting with Plotly
# Create a new Plotly subplot figure
fig = make_subplots()

# Add the energy consumption trace
fig.add_trace(
    go.Scatter(
        x=selected_consumption[STARTTIME].dt.strftime('%H:%M'), 
        y=selected_consumption[CONSUMPTION],
        mode='lines',
        name='Total Consumption',
        fill='tozeroy'
    )
)

# Add the renewable energy production trace
fig.add_trace(
    go.Scatter(
        x=selected_production[STARTTIME].dt.strftime('%H:%M'),
        y=selected_production['Total Production'],
        mode='lines',
        name='Total Renewable Production',
        fill='tozeroy'
    )
)


fig.update_layout(
    title=f'Energy Production and Consumption on {selected_date}',
    xaxis=dict(title='Time (hours)'),
    yaxis=dict(title='Energy (MWh)'),
    showlegend=True
)


# Show the plot using st.plotly_chart
fig.show()
#st.plotly_chart(fig)

#-------------------------------Dunkelflaute----------------------------------------------------------------------------------------


installed_power_dict = {
    2020: 122603,
    2021: 129551,
    2022: 133808
}

def find_dark_lulls(selected_date, production_df, installed_power_dict, count_dict):
    # Get the year of the selected date
    year = selected_date.year
    
    # Installed power for the corresponding year
    installed_power = installed_power_dict.get(year, None)
    
    if installed_power is None:
        print(f"No installed power found for the year {year}.")
        return None
    
    # Filter data for the selected date
    selected_production = production_df[production_df[DATE] == selected_date]
    
    # Sum the renewable energy production for the selected date
    total_renewable_production_selected_date = selected_production[columns_to_clean].sum(axis=1).sum()
    
    # Compare with installed power for different thresholds
    threshold_10_percent = installed_power * 0.1
    threshold_20_percent = installed_power * 0.2
    
    if total_renewable_production_selected_date/24 < threshold_10_percent:
        count_dict["up to 10%"].append(selected_date)
    elif total_renewable_production_selected_date/24 < threshold_20_percent:
        count_dict["up to 20%"].append(selected_date)
    else:
        return None

def find_dark_lulls_for_years(production_df, installed_power_dict):
    # Loop through all days in the years 2020 to 2022
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2022, 12, 31)

    dark_lulls_dict = {"up to 10%": [], "up to 20%": []}
    current_date = start_date
    
    while current_date <= end_date:
        find_dark_lulls(current_date, production_df, installed_power_dict, dark_lulls_dict)
        current_date += pd.DateOffset(days=1)
    
    # Sort lists by date
    for label, days_list in dark_lulls_dict.items():
        dark_lulls_dict[label] = sorted(days_list)
    
    # Display the sorted lists
    print("\nList of days up to 10%:")
    for day in dark_lulls_dict["up to 10%"]:
        print(day.strftime('%d.%m.%Y'))

    print("\nList of days up to 20%:")
    for day in dark_lulls_dict["up to 20%"]:
        print(day.strftime('%d.%m.%Y'))
    
    print("\nNumber of days up to 10%:", len(dark_lulls_dict["up to 10%"]))
    print("Number of days up to 20%:", len(dark_lulls_dict["up to 20%"]))


find_dark_lulls_for_years(production_df, installed_power_dict)



#--------------------------------------------------------------------------
# Abfrage datum
while True:
    selected_date_str = input("Bitte geben Sie ein Datum IM JAHR 2030 im Format TT.MM.JJJJ ein: ")
    selected_date = datetime.strptime(selected_date_str, "%d.%m.%Y")

    if selected_date.year == 2030:
        break
    else:
        print("Bitte geben Sie ein Datum aus dem Jahr 2030 an.")

# Define a dataframe of the production of 2022
production_2022df = production_df[production_df[DATE].dt.year == 2022]
prognoseErzeugung2030df = production_2022df.copy()
prognoseErzeugung2030df['Datum'] = prognoseErzeugung2030df['Datum'].map(lambda x: x.replace(year=2030))

# Define a dataframe of the consumption of 2022
consumption_2022df = consumption_df[consumption_df[DATE].dt.year == 2022]
prognoseVerbrauch2030df = consumption_2022df.copy()
prognoseVerbrauch2030df['Datum'] = prognoseVerbrauch2030df['Datum'].map(lambda x: x.replace(year=2030))

# CHANGE LATER Define the factors Verbrauch 2022 to 2030 von Bjarne Noah
# Verbrauch2022_2030_factor = 1.157  # assuming consumption will increase by 10% from 2022 compared to 2030


def scale_2030_factorsConsumption(df, Verbrauch2022_2030_factor):
    df_copy = df.copy()
    df_copy[CONSUMPTION] *= Verbrauch2022_2030_factor
    return df_copy


'''
#LATER put in Verbrauch 2030 fron noah&bjarne
prognoseVerbrauch2030df = energyConsumption(consumption_df)
'''
# Define the factors
# müssen noch angepasst werden
windonshore_2030_factor = 2.03563  # assuming Wind Onshore will increase by 203% from 2022 compared to 2030
windoffshore_2030_factor = 3.76979  # assuming Wind Offshore will 376% increase
pv_2030_factor = 3.5593  # assuming PV will increase by 350%


def scale_2030_factors(df, windonshore_factor, windoffshore_factor, pv_factor):
    df_copy = df.copy()
    df_copy[WIND_ONSHORE] *= windonshore_factor
    df_copy[WIND_OFFSHORE] *= windoffshore_factor
    df_copy[PHOTOVOLTAIC] *= pv_factor
    df_copy['Total Production'] = df_copy[columns_to_clean].sum(axis=1)
    return df_copy


# Scale the data by the factors
scaled_production_df = scale_2030_factors(prognoseErzeugung2030df, windonshore_2030_factor, windoffshore_2030_factor,
                                          pv_2030_factor)

# Filter the data for the selected date
scaled_selected_production_df = scaled_production_df[scaled_production_df[DATE] == selected_date]

verbrauch2030df = energyConsumption(consumption_df)
# scaled_consumption_df = scale_2030_factorsConsumption(prognoseVerbrauch2030df, Verbrauch2022_2030_factor)

# Filter the data for the selected date
# scaled_selected_consumption_df = scaled_consumption_df[scaled_consumption_df[DATE] == selected_date]

### subplot only consumption 2030

## Plotly daily production and consumption 2030

selected_consumption2030df = verbrauch2030df[verbrauch2030df[DATE] == selected_date]
scaled_selected_production_df = scaled_selected_production_df[scaled_selected_production_df[DATE] == selected_date]

fig = make_subplots()

# Add the energy consumption trace
fig.add_trace(
    go.Scatter(
        x=selected_consumption2030df[STARTTIME].dt.strftime('%H:%M'),
        y=selected_consumption2030df['Verbrauch [MWh]'],
        mode='lines',
        name='Total Consumption',
        fill='tozeroy'
    )
)

# Add the renewable energy production trace
fig.add_trace(
    go.Scatter(
        x=scaled_selected_production_df[STARTTIME].dt.strftime('%H:%M'),
        y=scaled_selected_production_df['Total Production'],
        mode='lines',
        name='Total Renewable Production',
        fill='tozeroy'
    )
)

fig.update_layout(
    title=f'Energy Production and Consumption on {selected_date}',
    xaxis=dict(title='Time (hours)'),
    yaxis=dict(title='Energy (MWh)'),
    showlegend=True
)

fig.show()

# code to do 2030 quarter hours
total_scaled_renewable_production = scaled_production_df[columns_to_clean].sum(axis=1)
total_consumption = verbrauch2030df['Verbrauch [MWh]']

# Berechnung der prozentualen Anteile der erneuerbaren Energieerzeugung am Gesamtverbrauch
percent_renewable = total_scaled_renewable_production / total_consumption * 100

counts, intervals = np.histogram(percent_renewable, bins=np.arange(0, 330, 1))  # Use NumPy to calculate the histogram of the percentage distribution

x = intervals[:-1]  # Define the x-axis values as the bin edges
labels = [f'{i}%' for i in range(0, 330, 1)]  # Create labels for x-axis ticks (von 0 bis 111 in Einzelnschritten)

fig = go.Figure(data=[go.Bar(x=x, y=counts)])  # Create a bar chart using Plotly
fig.update_layout(
    xaxis=dict(tickmode='array', tickvals=list(range(0, 330, 5)), ticktext=labels[::5]))  # X-axis label settings

# Title and axis labels settings
fig.update_layout(title='Anzahl der Viertelstunden im Jahren 2030 mit 0-330 % EE-Anteil',
                  xaxis_title='Prozentsatz erneuerbarer Energie',
                  yaxis_title='Anzahl der Viertelstunden')

fig.show()

# how many quarter hours are in scaled_production_df
print("soviele VS sind in scaled_production_df:")
print(len(scaled_selected_production_df))
print("Viertelstunden aus 2030 expected 35040 hier: ")




# Berechnen Sie den prozentualen Anteil der erneuerbaren Energieerzeugung am Verbrauch
prozentualerAnteil = total_scaled_renewable_production / verbrauch2030df['Verbrauch [MWh]'] * 100
print(percent_renewable)

# Initialisieren Sie eine Liste für die Datensätze
data = []

# Iterieren Sie über die Prozentsätze von 0 bis 100
for i in range(301):
    # Zählen Sie die Viertelstunden über oder gleich dem Prozentsatz
    anzahlViertelstundenProzent = len(percent_renewable[percent_renewable >= i])

    # Fügen Sie einen Datensatz zum Speichern in die Liste hinzu
    data.append({'Prozentsatz': i, 'Anzahl_Viertelstunden': anzahlViertelstundenProzent})

# Erstellen Sie ein DataFrame aus der Liste
result_df = pd.DataFrame(data)

# Drucken Sie das erstellte DataFrame
print(result_df)

fig = go.Figure()

# Fügen Sie einen Balken für die Anzahl der Viertelstunden für jeden Prozentsatz hinzu
fig.add_trace(go.Bar(x=result_df['Prozentsatz'], y=result_df['Anzahl_Viertelstunden']))

# Aktualisieren Sie das Layout für Titel und Achsenbeschriftungen
fig.update_layout(
    title='Anzahl der Viertelstunden mit erneuerbarer Energieerzeugung über oder gleich dem Verbrauch',
    xaxis=dict(title='Prozentsatz erneuerbarer Energie'),
    yaxis=dict(title='Anzahl der Viertelstunden')
)

# Zeigen Sie den Plot an
fig.show()

#Speicherberrechnung von Timo hier Leistung

def powerOfStorage(verbrauch_df, erzeugung_df,prozent):
    # Kopien der DataFrames erstellen, um den Originalinhalt nicht zu verändern
    verbrauch_copy = verbrauch_df.copy()
    erzeugung_copy = erzeugung_df.copy()
    
    # Setzen des Datums als Index für die einfache Berechnung der Differenz
    erzeugung_copy.set_index('Datum', inplace=True)
    verbrauch_copy.set_index('Datum', inplace=True)
    
    # Berechnung der Differenz zwischen Verbrauch und Erzeugung auf Viertelstundenbasis
    differenz =(verbrauch_copy['Verbrauch [MWh]'] *prozent)- erzeugung_copy['Total Production']
    
    # Neuer DataFrame für die Differenz erstellen
    differenz_data = pd.DataFrame({'Differenz': differenz})
    
    # Sortieren des DataFrames nach der Spalte 'Differenz'
    differenz_data_sorted = differenz_data.sort_values(by='Differenz', ascending=False)
    
   
    
    # Mittelwert der ersten 100 größten Differenzen berechnen
    mean_top_100 = differenz_data_sorted.head(100)['Differenz'].mean()
    power_in_GW =mean_top_100 /(0.25*1000) #Umrechnung des Mittelwerts in GW und Leistung
    
    return differenz_data_sorted,power_in_GW

# Verwendung der Funktion mit den entsprechenden DataFrames verbrauch2030df und scaled_production_df
result_differenz_sorted_80, power_in_GW_80 = powerOfStorage(verbrauch2030df, scaled_production_df,0.8)

# Ausgabe des sortierten DataFrames
print("Sortiertes DataFrame nach Differenz:")
print(result_differenz_sorted_80)

# Ausgabe des Mittelwerts der ersten 100 größten Differenzen
print("Leistung der Speicher für 80% Dekung in GW:", power_in_GW_80)

# Verwendung der Funktion mit den entsprechenden DataFrames verbrauch2030df und scaled_production_df
result_differenz_sorted_90, power_in_GW_90 = powerOfStorage(verbrauch2030df, scaled_production_df,0.9)

# Ausgabe des sortierten DataFrames
print("Sortiertes DataFrame nach Differenz:")
print(result_differenz_sorted_90)

# Ausgabe des Mittelwerts der ersten 100 größten Differenzen
print("Leistung der Speicher für 90% Dekung in GW:", power_in_GW_90)

# Verwendung der Funktion mit den entsprechenden DataFrames verbrauch2030df und scaled_production_df
result_differenz_sorted_100, power_in_GW_100 = powerOfStorage(verbrauch2030df, scaled_production_df,1)

# Ausgabe des sortierten DataFrames
print("Sortiertes DataFrame nach Differenz:")
print(result_differenz_sorted_100)

# Ausgabe des Mittelwerts der ersten 100 größten Differenzen
print("Leistung der Speicher für 100% Dekung in GW:", power_in_GW_100) 

#Benötigte Leistung für den Überschuss

def powerOfStorageforsurplus(verbrauch_df, erzeugung_df,prozent):
    # Kopien der DataFrames erstellen, um den Originalinhalt nicht zu verändern
    verbrauch_copy = verbrauch_df.copy()
    erzeugung_copy = erzeugung_df.copy()
    
    # Setzen des Datums als Index für die einfache Berechnung der Differenz
    erzeugung_copy.set_index('Datum', inplace=True)
    verbrauch_copy.set_index('Datum', inplace=True)
    
    # Berechnung der Differenz zwischen Verbrauch und Erzeugung auf Viertelstundenbasis
    differenz =erzeugung_copy['Total Production']-(verbrauch_copy['Verbrauch [MWh]']*prozent)
    
    # Neuer DataFrame für die Differenz erstellen
    differenz_data = pd.DataFrame({'Differenz': differenz})
    
    # Sortieren des DataFrames nach der Spalte 'Differenz'
    differenz_data_sorted = differenz_data.sort_values(by='Differenz', ascending=False)
    
   
    
    # Mittelwert der ersten 100 größten Differenzen berechnen
    mean_top_100 = differenz_data_sorted.head(100)['Differenz'].mean()
    power_in_GW =mean_top_100 /(0.25*1000) #Umrechnung des Mittelwerts in GW und Leistung
    
    return differenz_data_sorted,power_in_GW 

# Verwendung der Funktion mit den entsprechenden DataFrames verbrauch2030df und scaled_production_df
result_differenz_sorted_surplus, power_in_GW_surplus = powerOfStorageforsurplus(verbrauch2030df, scaled_production_df,0.8)

# Ausgabe des sortierten DataFrames
print("Sortiertes DataFrame nach Differenz:")
print(result_differenz_sorted_surplus)

# Ausgabe des Mittelwerts der ersten 100 größten Differenzen
print("Leistung der Speicher für die Überschussaufnahme in GW:", power_in_GW_surplus)

#Ermittlung der größten Kapazität der Speicher anhand der Längsten periode wo die Erzeugung unter dem Verbrauch ist


def calculate_storage_capacity(verbrauch_df, erzeugung_df,prozent):
    verbrauch_copy = verbrauch_df.copy()
    erzeugung_copy = erzeugung_df.copy()

    # Setze das Datum als Index für die einfache Berechnung der Differenz
    erzeugung_copy.set_index('Datum', inplace=True)
    verbrauch_copy.set_index('Datum', inplace=True)

    differenz = (verbrauch_copy['Verbrauch [MWh]'])*prozent - erzeugung_copy['Total Production']
    differenz_data = pd.DataFrame({'Differenz': differenz})

    max_period = pd.Timestamp.now()
    max_sum = 0

    start_date = pd.Timestamp.now()
    end_date = pd.Timestamp.now()

    current_start = pd.Timestamp.now()
    current_sum = 0

    for index, value in differenz.items():
        if value >= 0:
            if current_sum <= 0:
                current_start = index
                current_sum = 0

            current_sum += value

            if current_sum > max_sum:
                max_sum = current_sum
                start_date = current_start
                end_date = index

        else:
            current_sum = 0

    return start_date, end_date, max_sum

start_date, end_date, max_sum = calculate_storage_capacity(verbrauch2030df, scaled_production_df,0.8)

# Ausgabe der Ergebnisse im Hauptprogramm

print("Startdatum der längsten Zeitspanne mit positiven Differenzen:", start_date)
print("Enddatum der längsten Zeitspanne mit positiven Differenzen:", end_date)
print("Summe der längsten Zeitspanne mit positiven Differenzen:", max_sum)




def capacity(verbrauch_df, erzeugung_df, prozent, start_capacity):
    verbrauch_copy = verbrauch_df.copy()
    erzeugung_copy = erzeugung_df.copy()
    capacity_value = 0  # Verwende den übergebenen Startwert der Kapazität
    enrgieSurPlus=0
    efficencie=0.9 #Mittel von Pump und Batteriespeicher

    # Setze das Datum als Index für die einfache Berechnung der Differenz
    erzeugung_copy.set_index('Datum', inplace=True)
    verbrauch_copy.set_index('Datum', inplace=True)

    # Berechne die Differenz zwischen Verbrauch und Erzeugung auf Viertelstundenbasis
    differenz = (verbrauch_copy['Verbrauch [MWh]']) * prozent - erzeugung_copy['Total Production']
    differenz_data = pd.DataFrame({'Differenz': differenz})

    total_consumption = verbrauch_copy['Verbrauch [MWh]'].sum()
    total_production = erzeugung_copy['Total Production'].sum()
    
    # Berechne den prozentualen Anteil des Verbrauchs zur Erzeugung auf Viertelstundenbasis
    percentage = (total_consumption / total_production)
    
    if percentage<=prozent:
        percentage=percentage*100
        print(f"Die Erzeugung kann den angegebenen verbrauch nicht decken {percentage}%")
        return 0,0
    else:
        while capacity_value == 0:
         start_capacity=start_capacity+1000000
         capacity_value =start_capacity
         
         energieSurPlus=0

         # Iteriere über die Differenzen
         for index, value in differenz_data.iterrows():
         # Wenn die Differenz positiv ist, entnehme der Kapazität
          if value['Differenz'] > 0:
             # Überprüfe, ob die Kapazität leer wird
             if capacity_value - value['Differenz'] < 0:
                # Berechne den verbleibenden positiven Wert, der noch entnommen werden kann
                remaining_capacity = capacity_value
                capacity_value=0
                print(f"Speicher ist leer, es konnten nur {capacity_value} MWh entnommen werden.")
                break
                
                
             else:
                capacity_value -= value['Differenz']
                #print(f"Entnehme {value['Differenz']} MWh aus dem Speicher. Aktuelle Kapazität: {capacity_value} MWh")

         # Wenn die Differenz negativ ist, füge der Kapazität hinzu
          elif value['Differenz'] < 0:
             # Überprüfe, ob mehr eingespeichert werden kann als der Wert der Kapazität
             if capacity_value + (abs(value['Differenz'])*efficencie) > start_capacity:
                #print("Es kann nicht mehr eingespeichert werden als die verfügbare Kapazität.")
                energieSurPlus=capacity_value + abs(value['Differenz'])*efficencie-start_capacity

                capacity_value = start_capacity
             else:
                capacity_value -= (value['Differenz']*efficencie)
               # print(f"Füge {abs(value['Differenz'])} MWh dem Speicher hinzu. Aktuelle Kapazität: {capacity_value} MWh")
         
        return capacity_value, start_capacity,energieSurPlus
   

capacity_value, capacity_value_start,energieSurPlus = capacity(verbrauch2030df, scaled_production_df, 0.8, 10000000)

print('capacity value' + str(capacity_value))
print('capacity_value_start' + str(capacity_value_start))
print('energieSurPlus' + str(energieSurPlus))

#Evtuell auch gut geeignet
def capacity2(verbrauch_df, erzeugung_df, prozent):
    verbrauch_copy = verbrauch_df.copy()
    erzeugung_copy = erzeugung_df.copy()
    capacity_value = 0  # Verwende den übergebenen Startwert der Kapazität
    maxnegativ=0
    maxpositiv=0

    # Setze das Datum als Index für die einfache Berechnung der Differenz
    erzeugung_copy.set_index('Datum', inplace=True)
    verbrauch_copy.set_index('Datum', inplace=True)

    # Berechne die Differenz zwischen Verbrauch und Erzeugung auf Viertelstundenbasis
    differenz = (verbrauch_copy['Verbrauch [MWh]']) * prozent - erzeugung_copy['Total Production']
    differenz_data = pd.DataFrame({'Differenz': differenz})

    
    
    for index, value in differenz_data.iterrows():
        # Wenn die Differenz positiv ist, entnehme der Kapazität
        if value['Differenz'] > 0:
            # Überprüfe, ob die Kapazität leer wird
            if capacity_value - value['Differenz'] <maxnegativ :
                # Berechne den verbleibenden positiven Wert, der noch entnommen werden kann
                
                capacity_value -= value['Differenz']
                maxnegativ=capacity_value
                               
            else:
                capacity_value -= value['Differenz']
                #print(f"Entnehme {value['Differenz']} MWh aus dem Speicher. Aktuelle Kapazität: {capacity_value} MWh")

        # Wenn die Differenz negativ ist, füge der Kapazität hinzu
        elif value['Differenz'] < 0:
         if capacity_value - value['Differenz'] >maxpositiv :
            
          capacity_value -= value['Differenz']
          maxpositiv=capacity_value
         else:
             capacity_value -= value['Differenz']
             
              

    return maxnegativ,maxpositiv

#maxnegativ,maxpositiv = capacity2(verbrauch2030df, scaled_production_df, 0.8)
#print(maxnegativ)
#print(maxpositiv)

def investmentcost(capacity_needed):   #Eventuell noch prozente von Speicherarten hinzufügen
 capacity_in_germany=0  
 cost_of_Battery=100 #Einheit sind Euro/kWh

 capacity_for_expension=capacity_needed-capacity_in_germany

 price=(cost_of_Battery*capacity_for_expension)/(1000000) #Price in Bilion

 print(f"Der Preis in Milliarden beträgt:{price}")


investmentcost(capacity_value_start)
