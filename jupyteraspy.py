# %%
import pandas as pd
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# %% [markdown]
# **Author: K.Bodrova:**

# %%
# Function to read and clean data from CSV files
def read_and_clean_data(file_production, file_consumption):
    
    production_df = pd.read_csv(file_production, delimiter=';')                           # Read data from CSV files
    consumption_df = pd.read_csv(file_consumption, delimiter=';')

    production_df['Date'] = pd.to_datetime(production_df['Datum'], format='%d.%m.%Y')     # Convert date columns to DateTime objects
    production_df['Starttime'] = pd.to_datetime(production_df['Anfang'], format='%H:%M')
    consumption_df['Date'] = pd.to_datetime(consumption_df['Datum'], format='%d.%m.%Y')
    consumption_df['Starttime'] = pd.to_datetime(consumption_df['Anfang'], format='%H:%M')
    
    # Clean data formats for renewable energies
    production_df['Biomass'] = production_df['Biomasse [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").replace('-', 0).astype(float)
    production_df['Hydroelectric'] = production_df['Wasserkraft [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").replace('-', 0).astype(float)
    production_df['Wind Offshore'] = production_df['Wind Offshore [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").replace('-', 0).astype(float)
    production_df['Wind Onshore'] = production_df['Wind Onshore [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").replace('-', 0).astype(float)
    production_df['Photovoltaic'] = production_df['Photovoltaik [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").replace('-', 0).astype(float)
    production_df['Other Renewable'] = production_df['Sonstige Erneuerbare [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").replace('-', 0).astype(float)

    # Clean data formats for total consumption
    consumption_df['Consumption'] = consumption_df['Gesamt (Netzlast) [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)

    # Create a new column for total production
    production_df['Total Production'] = production_df[['Biomass', 'Hydroelectric', 'Wind Offshore', 'Wind Onshore', 'Photovoltaic', 'Other Renewable']].sum(axis=1)

    # Gruppierung der Produktionsdaten nach Jahr und Summierung der erneuerbaren Energietypen
    production_by_type_and_year = production_df.groupby(production_df['Date'].dt.year)[['Biomass', 'Hydroelectric', 'Wind Offshore', 'Wind Onshore', 'Photovoltaic', 'Other Renewable']].sum()
 
    pd.options.display.float_format = '{:.2f}'.format  # Set Pandas to display floating-point numbers with two decimal places

    data_by_year = {}  # Aggregation der Daten nach Jahren und Speicherung in einem Dictionary

    for year, data in production_df.groupby(production_df['Date'].dt.year):
        production_data = data[['Biomass', 'Hydroelectric', 'Wind Offshore', 'Wind Onshore', 'Photovoltaic', 'Other Renewable']].sum()
        consumption_data = consumption_df[consumption_df['Date'].dt.year == year]['Consumption']
        total_consumption = consumption_data.sum()
        data_by_year[year] = {
            'Production': production_data.sum(),
            'Consumption': total_consumption,
            'Biomass': production_data['Biomass'],
            'Hydroelectric': production_data['Hydroelectric'],
            'Wind Offshore': production_data['Wind Offshore'],
            'Wind Onshore': production_data['Wind Onshore'],
            'Photovoltaic': production_data['Photovoltaic'],
            'Other Renewable': production_data['Other Renewable']
        }

    total_renewable_production = production_df[['Biomass', 'Hydroelectric', 'Wind Offshore', 'Wind Onshore', 'Photovoltaic', 'Other Renewable']].sum(axis=1)
    total_consumption = consumption_df['Consumption']

    return production_df, consumption_df, total_renewable_production, total_consumption, data_by_year

# %%
def read_load_profile(file_path):
    # Read the Excel file
    load_profile_df = pd.read_excel(file_path, skiprows=8)
    
    # Rename the columns for clarity
    load_profile_df.columns = ['Time', 'Weekday_Summer', 'Saturday_Summer', 'Sunday_Summer', 'Weekday_Winter', 'Saturday_Winter', 'Sunday_Winter']
    
    # Define a function to replace "24:00:00" with "00:00:00"
    def replace_24_with_00(time_str):   
        return time_str.replace('24:00:00', '00:00:00')

    # Apply the function to each value in the 'Time' column
    load_profile_df['Time'] = load_profile_df['Time'].apply(replace_24_with_00)

    # Remove leading and trailing whitespace
    load_profile_df['Time'] = load_profile_df['Time'].str.strip()

    # Convert the 'Time' column to a DateTime object
    load_profile_df['Time'] = pd.to_datetime(load_profile_df['Time'], format='%H:%M:%S').dt.time

    # Multiply all values (except 'Time') by 32*10^6
    cols_to_update = ['Weekday_Summer', 'Saturday_Summer', 'Sunday_Summer', 'Weekday_Winter', 'Saturday_Winter', 'Sunday_Winter']
    load_profile_df[cols_to_update] = load_profile_df[cols_to_update].applymap(lambda x: x * 32 * 10**6)
    
    return load_profile_df

# %% [markdown]
# **Authors: M.Lauterbach, K.Bodrova:**

# %%
installed_power_dict = {2020: 122603, 2021: 129551, 2022: 133808}

def find_dark_lulls_and_for_years(selected_date, production_df, dark_lulls_dict, columns_to_clean):
    
    year = selected_date.year  # Get the year of the selected date
    
    installed_power = installed_power_dict.get(year, None) # Installed power for the corresponding year
    
    if installed_power is None:
        print(f"No installed power found for the year {year}.")
        return None
    
    selected_production = production_df[production_df['Date'] == selected_date] # Filter data for the selected date
    
    total_renewable_production_selected_date = selected_production[columns_to_clean].sum(axis=1).sum() # Sum the renewable energy production for the selected date
    
    threshold_10_percent = installed_power * 0.1 # Compare with installed power for different thresholds
    threshold_20_percent = installed_power * 0.2
    
    if total_renewable_production_selected_date/24 < threshold_10_percent:
        dark_lulls_dict["up to 10%"].append(selected_date)
    elif total_renewable_production_selected_date/24 < threshold_20_percent:
        dark_lulls_dict["up to 20%"].append(selected_date)
    else:
        return None

def find_dark_lulls_for_years(production_df, columns_to_clean):
    start_date = datetime(2020, 1, 1)           # Loop through all days in the years 2020 to 2022
    end_date = datetime(2022, 12, 31)

    dark_lulls_dict = {"up to 10%": [], "up to 20%": []}
    current_date = start_date
    
    while current_date <= end_date:
        find_dark_lulls_and_for_years(current_date, production_df, dark_lulls_dict, columns_to_clean)
        current_date += pd.DateOffset(days=1)
    
    for label, days_list in dark_lulls_dict.items():    # Sort lists by date
        dark_lulls_dict[label] = sorted(days_list)
    
    print("\nList of days up to 10%:")                 # Display the sorted lists
    for day in dark_lulls_dict["up to 10%"]:
        print(day.strftime('%d.%m.%Y'))

    print("\nList of days up to 20%:")
    for day in dark_lulls_dict["up to 20%"]:
        print(day.strftime('%d.%m.%Y'))

    print("\nNumber of days up to 10%:", len(dark_lulls_dict["up to 10%"]))
    print("Number of days up to 20%:", len(dark_lulls_dict["up to 20%"]))

# %% [markdown]
# **Author: K.Bodrova:**

# %%
# Funktion zur Berechnung und Anzeige des Histogramms für erneuerbare Anteile
def calculate_and_display_renewable_shares_histogram(total_renewable_production, total_consumption):
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

# %% [markdown]
# **Function to plot energy consumption and production for a selected date.
#  Author: K. Bodrova, Diagram: O. Pearse-Danker:**

# %%
def plot_energy_consumption_and_production(production_df, consumption_df, columns_to_clean):
    selected_date_str = input("Enter the selected date (format: YYYY-MM-DD): ") # Ask the user to enter a date    
    try:
        selected_date = pd.to_datetime(selected_date_str)
    except ValueError:
        print("Invalid date format. Please use the format YYYY-MM-DD.")
        return

    selected_production = production_df[production_df['Date'] == selected_date]       # Filter data for the selected date
    selected_consumption = consumption_df[consumption_df['Date'] == selected_date]

    total_renewable_production_selected_date = selected_production[columns_to_clean].sum(axis=1).sum()
    print(f"Summe der erneuerbaren Energien am {selected_date}: {total_renewable_production_selected_date} MWh")

    fig = make_subplots()                 # Create a new Plotly subplot figure

    fig.add_trace(                        # Add the energy consumption trace
        go.Scatter(
            x=selected_consumption['Starttime'].dt.strftime('%H:%M'),
            y=selected_consumption['Consumption'],
            mode='lines',
            name='Total Consumption',
            fill='tozeroy'
        )
    )
 
    fig.add_trace(                        # Add the renewable energy production trace
        go.Scatter(
            x=selected_production['Starttime'].dt.strftime('%H:%M'),
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

    fig.show()                            # Display the figure

# %% [markdown]
# **Author: K.Bodrova:**

# %%
# Funktion zur Berechnung und Anzeige der aggregierten Daten pro Jahr
def calculate_and_display_data_by_year(data_by_year):
    for year, data in data_by_year.items():
        print(f"Jahr: {year}")
        print(f"Gesamte erneuerbare Energieproduktion: {data['Production']} MWh")
        print(f"Gesamtverbrauch: {data['Consumption']} MWh")
        print(f"Biomasse: {data['Biomass']} MWh")
        print(f"Wasserkraft: {data['Hydroelectric']} MWh")
        print(f"Wind Offshore: {data['Wind Offshore']} MWh")
        print(f"Wind Onshore: {data['Wind Onshore']} MWh")
        print(f"Photovoltaik: {data['Photovoltaic']} MWh")
        print(f"Andere erneuerbare Energien: {data['Other Renewable']} MWh")
        print()

# %%
def plot_energy_data(consumption_df, production_df, selected_date):
    fig = make_subplots()

    # Add the energy consumption trace
    fig.add_trace(
        go.Scatter(
            x=consumption_df['Starttime'].dt.strftime('%H:%M'),
            y=consumption_df['Verbrauch [MWh]'],
            mode='lines',
            name='Total Consumption',
            fill='tozeroy'
        )
    )

    # Add the renewable energy production trace
    fig.add_trace(
        go.Scatter(
            x=production_df['Starttime'].dt.strftime('%H:%M'),
            y=production_df['Total Production'],
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

# %%
def plot_renewable_percentage(scaled_production_df, verbrauch2030df):
    total_scaled_renewable_production = scaled_production_df[['Biomass', 'Hydroelectric', 'Wind Offshore', 'Wind Onshore', 'Photovoltaic', 'Other Renewable']].sum(axis=1)
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


    data = []

    for i in range(301):
        # Zählen die Viertelstunden über oder gleich dem Prozentsatz
        anzahlViertelstundenProzent = len(percent_renewable[percent_renewable >= i])
         # Fügen Sie einen Datensatz zum Speichern in die Liste hinzu
        data.append({'Prozentsatz': i, 'Anzahl_Viertelstunden': anzahlViertelstundenProzent})
    
    result_df = pd.DataFrame(data) # DataFrame erstellen
    
    fig = go.Figure()

    # Fügen einen Balken für die Anzahl der Viertelstunden für jeden Prozentsatz hinzu
    fig.add_trace(go.Bar(x=result_df['Prozentsatz'], y=result_df['Anzahl_Viertelstunden']))

    # Aktualisieren Sie das Layout für Titel und Achsenbeschriftungen
    fig.update_layout(
        title='Anzahl der Viertelstunden mit erneuerbarer Energieerzeugung über oder gleich dem Verbrauch',
        xaxis=dict(title='Prozentsatz erneuerbarer Energie'),
        yaxis=dict(title='Anzahl der Viertelstunden')
    )

    fig.show()

# %% [markdown]
# **Authors: L.Dorda, N.Clasen, B.Wolf:**

# %%
# Function to process and plot data for the year 2030
def process_and_plot_2030_dataGut(production_df, consumption_df, load_profile_df, selected_date):
    
    # POSITIVE SCENARIO Production based on 2020 and BMWK goals
    production_2020df = production_df[production_df['Date'].dt.year == 2020]
    prognoseErzeugung2030_positive_df = production_2020df.copy()
    #prognoseErzeugung2030_positive_df['Date'] = prognoseErzeugung2030_positive_df['Date'].map(lambda x: x.replace(year=2030))
    prognoseErzeugung2030_positive_df['Date'] = prognoseErzeugung2030_positive_df['Date'].map(
    lambda x: x.replace(year=2030) if not (x.month == 2 and x.day == 29) else x.replace(month=2, day=28, year=2030))

    windonshore_2030_factor_2020_positive = 2.13589  # 
    windoffshore_2030_factor_2020_postive = 3.92721  #
    pv_2030_factor_2020_postive = 4.2361193  # assumig PV will increase by 423%

    def scale_2030_factors(df,windonshore_2030_factor_2020_positive,windoffshore_2030_factor_2020_postive,
                                          pv_2030_factor_2020_postive):
        df_copy = df.copy()
        df_copy['Wind Onshore'] *= windonshore_2030_factor_2020_positive
        df_copy['Wind Offshore'] *= windoffshore_2030_factor_2020_postive
        df_copy['Photovoltaic'] *= pv_2030_factor_2020_postive
        df_copy['Total Production'] = df_copy[['Biomass', 'Hydroelectric', 'Wind Offshore', 'Wind Onshore', 'Photovoltaic', 'Other Renewable']].sum(axis=1)
        return df_copy

    # Scale the data by the factors
    scaled_production_df = scale_2030_factors(prognoseErzeugung2030_positive_df, windonshore_2030_factor_2020_positive,windoffshore_2030_factor_2020_postive,
                                          pv_2030_factor_2020_postive)

    #_________________________________________________________________________________________________________
    # Filter the data for the selected date
    scaled_selected_production_df = scaled_production_df[scaled_production_df['Date'] == selected_date]

    verbrauch2030df = energyConsumption(consumption_df)

    selected_consumption2030df = verbrauch2030df[verbrauch2030df['Date'] == selected_date]
    scaled_selected_production_df = scaled_selected_production_df[scaled_selected_production_df['Date'] == selected_date]

    plot_energy_data(selected_consumption2030df, scaled_selected_production_df, selected_date)
    plot_renewable_percentage(scaled_production_df, verbrauch2030df)

# Funktion zur Berechnung und Anzeige der aggregierten Daten pro Jahr
# Author: Bjarne, Noah
def energyConsumption(consumption_df):
    wärmepumpeHochrechnung2030 = wärmepumpe()
    eMobilitätHochrechnung2030 = eMobilität()

    verbrauch2022df = consumption_df[consumption_df['Date'].dt.year == 2020]
    prognose2030df = verbrauch2022df.copy()
    faktor = faktorRechnung(verbrauch2022df, wärmepumpeHochrechnung2030, eMobilitätHochrechnung2030)
    print("Verbr df:", prognose2030df)
    print("Faktor: ", faktor)
    # Change the year in 'Datum' column to 2030
    prognose2030df['Date'] = prognose2030df['Date'].map(lambda x: x.replace(year=2030) if not (x.month == 2 and x.day == 29) else x.replace(month=2, day=28, year=2030))

    prognose2030df['Verbrauch [MWh]'] = prognose2030df['Consumption'] * faktor

    combined_df = pd.concat([verbrauch2022df[['Starttime', 'Consumption']], prognose2030df[['Verbrauch [MWh]']]], axis=1)
    print("Verbrauch 2030:", prognose2030df['Verbrauch [MWh]'].sum()/1000 , "TWhhusp\n")
    print("Consumption 2022:", prognose2030df['Consumption'].sum()/1000 , "TWh\n")

    return prognose2030df

def wärmepumpe():
    highScenario = 500000
    lowScenario = 236000
    middleScenario = 368000
    wärmepumpeAnzahl2030 = lowScenario * (2030 - 2023)  # 500k pro Jahr bis 2023

    heizstunden = 2000
    nennleistung = 15  # 15kW
    luftWasserVerhältnis = 206 / 236
    erdwärmeVerhältnis = 30 / 236
    luftWasserJAZ = 3.1
    erdwärmeJAZ = 4.1

    # Berechnung der einzelnen Pumpe
    luftWasserVerbrauch = wärmepumpeVerbrauchImJahr(heizstunden, nennleistung, luftWasserJAZ)  # in kW/h
    erdwärmeVerbrauch = wärmepumpeVerbrauchImJahr(heizstunden, nennleistung, erdwärmeJAZ)  # in kW/h

    luftWasserVerhältnisAnzahl = verhältnisAnzahl(wärmepumpeAnzahl2030, luftWasserVerhältnis)
    erdwärmeVerhältnisAnzahl = verhältnisAnzahl(wärmepumpeAnzahl2030, erdwärmeVerhältnis)

    return luftWasserVerbrauch * luftWasserVerhältnisAnzahl + erdwärmeVerbrauch * erdwärmeVerhältnisAnzahl  # kWh

# berechnung des Verbrauchs einer Wärmepumpe im Jahr
def wärmepumpeVerbrauchImJahr(heizstunden, nennleistung, jaz): 
    return (heizstunden * nennleistung) / jaz # (Heizstunden * Nennleistung) / JAZ = Stromverbrauch pro Jahr

def verhältnisAnzahl(wärmepumpeAnzahl2030, verhältnis):
    return wärmepumpeAnzahl2030 * verhältnis


def eMobilität():
    highECars = 15000000
    lowECars = 8000000
    middleECars = 11500000

    eMobilität2030 = lowECars  # 15mio bis 20230
    eMobilitätBisher = 1307901  # 1.3 mio
    verbrauchPro100km = 21  # 21kWh
    kilometerProJahr = 15000  # 15.000km

    eMobilitätVerbrauch = (verbrauchPro100km / 100) * kilometerProJahr  # kWh

    return (eMobilität2030 - eMobilitätBisher) * eMobilitätVerbrauch

def faktorRechnung(verbrauch2022df, wärmepumpeHochrechnung2030, eMobilitätHochrechnung2030):
    gesamtVerbrauch2022 = (otherFactors(wärmepumpeHochrechnung2030, verbrauch2022df))*1000000000 + 504515946000 # mal1000 weil MWh -> kWh
    return (gesamtVerbrauch2022 + wärmepumpeHochrechnung2030 + eMobilitätHochrechnung2030) / (504515946000) #ges Verbrauch 2021

def prognoseRechnung(verbrauch2022df, faktor):
    verbrauch2030df = verbrauch2022df['Verbrauch [kWh]'] * faktor
    return verbrauch2030df

def otherFactors(wärmepumpeHochrechnung2030, verbrauch2022df):
    indHigh = (wärmepumpeHochrechnung2030*(1+3/7))*(72/26)
    indLow = verbrauch2022df['Consumption'].sum()*0.45*0.879/1000000
    indMiddle = 0

    # positive Faktoren
    railway = 5  # TWh
    powerNetLoss = 1
    industry = indLow

    # negative Faktoren
    efficiency = 51
    other = 6

    return railway  + powerNetLoss - efficiency - other + industry/1000000000

# %%
def process_and_plot_2030_dataSchlecht(production_df, consumption_df, load_profile_df, selected_date):
    
    # Realistisches Ausbau (based on frauenhofer) Szenario 2030 basierend auf 2022 Wetter (mittleres Wetter) ((2021 wäre schlechtes Wetter))
    production_2022df = production_df[production_df['Date'].dt.year == 2022]
    prognoseErzeugung2030_realistic_2022_df = production_2022df.copy()
    prognoseErzeugung2030_realistic_2022_df['Date'] = prognoseErzeugung2030_realistic_2022_df['Date'].map(lambda x: x.replace(year=2030) if not (x.month == 2 and x.day == 29) else x.replace(month=2, day=28, year=2030))

    windonshore_2030_factor_2022_realistic = 1.2921  # 
    windoffshore_2030_factor_2022_realistic = 2.13621  # 
    pv_2030_factor_2022_realistic = 1.821041  # assumig PV will increase by 182%

    def scale_2030_factors(df,windonshore_2030_factor_2022_realistic,windoffshore_2030_factor_2022_realistic,
                                          pv_2030_factor_2022_realistic):
        df_copy = df.copy()
        df_copy['Wind Onshore'] *= windonshore_2030_factor_2022_realistic
        df_copy['Wind Offshore'] *= windoffshore_2030_factor_2022_realistic
        df_copy['Photovoltaic'] *= pv_2030_factor_2022_realistic
        df_copy['Total Production'] = df_copy[['Biomass', 'Hydroelectric', 'Wind Offshore', 'Wind Onshore', 'Photovoltaic', 'Other Renewable']].sum(axis=1)
        return df_copy

    # Scale the data by the factors
    scaled_production_df = scale_2030_factors(prognoseErzeugung2030_realistic_2022_df, windonshore_2030_factor_2022_realistic,windoffshore_2030_factor_2022_realistic,
                                          pv_2030_factor_2022_realistic)

    # Filter the data for the selected date
    scaled_selected_production_df = scaled_production_df[scaled_production_df['Date'] == selected_date]

    verbrauch2030df = energyConsumption1(consumption_df)

    selected_consumption2030df = verbrauch2030df[verbrauch2030df['Date'] == selected_date]
    scaled_selected_production_df = scaled_selected_production_df[scaled_selected_production_df['Date'] == selected_date]

    plot_energy_data(selected_consumption2030df, scaled_selected_production_df, selected_date)
    plot_renewable_percentage(scaled_production_df, verbrauch2030df)

# Funktion zur Berechnung und Anzeige der aggregierten Daten pro Jahr
# Author: Bjarne, Noah
def energyConsumption1(consumption_df):
    wärmepumpeHochrechnung2030 = wärmepumpe1()
    eMobilitätHochrechnung2030 = eMobilität1()

    verbrauch2022df = consumption_df[consumption_df['Date'].dt.year == 2022]
    prognose2030df = verbrauch2022df.copy()
    faktor = faktorRechnung1(verbrauch2022df, wärmepumpeHochrechnung2030, eMobilitätHochrechnung2030)

    prognose2030df['Date'] = prognose2030df['Date'].map(lambda x: x.replace(year=2030) if not (x.month == 2 and x.day == 29) else x.replace(month=2, day=28, year=2030))

    prognose2030df['Verbrauch [MWh]'] = prognose2030df['Consumption'] * faktor

    combined_df = pd.concat([verbrauch2022df[['Starttime', 'Consumption']], prognose2030df[['Verbrauch [MWh]']]], axis=1)

    return prognose2030df

def wärmepumpe1():
    highScenario = 500000
    lowScenario = 236000
    middleScenario = 368000
    wärmepumpeAnzahl2030 = highScenario * (2030 - 2023)  # 500k pro Jahr bis 2023

    heizstunden = 2000
    nennleistung = 15  # 15kW
    luftWasserVerhältnis = 206 / 236
    erdwärmeVerhältnis = 30 / 236
    luftWasserJAZ = 3.1
    erdwärmeJAZ = 4.1

    # Berechnung der einzelnen Pumpe
    luftWasserVerbrauch = wärmepumpeVerbrauchImJahr1(heizstunden, nennleistung, luftWasserJAZ)  # in kW/h
    erdwärmeVerbrauch = wärmepumpeVerbrauchImJahr1(heizstunden, nennleistung, erdwärmeJAZ)  # in kW/h

    luftWasserVerhältnisAnzahl = verhältnisAnzahl1(wärmepumpeAnzahl2030, luftWasserVerhältnis)
    erdwärmeVerhältnisAnzahl = verhältnisAnzahl1(wärmepumpeAnzahl2030, erdwärmeVerhältnis)

    return luftWasserVerbrauch * luftWasserVerhältnisAnzahl + erdwärmeVerbrauch * erdwärmeVerhältnisAnzahl  # kWh

# berechnung des Verbrauchs einer Wärmepumpe im Jahr
def wärmepumpeVerbrauchImJahr1(heizstunden, nennleistung, jaz): 
    return (heizstunden * nennleistung) / jaz # (Heizstunden * Nennleistung) / JAZ = Stromverbrauch pro Jahr

def verhältnisAnzahl1(wärmepumpeAnzahl2030, verhältnis):
    return wärmepumpeAnzahl2030 * verhältnis

def eMobilität1():
    highECars = 15000000
    lowECars = 8000000
    middleECars = 11500000

    eMobilität2030 = highECars  # 15mio bis 20230
    eMobilitätBisher = 1307901  # 1.3 mio
    verbrauchPro100km = 21  # 21kWh
    kilometerProJahr = 15000  # 15.000km

    eMobilitätVerbrauch = (verbrauchPro100km / 100) * kilometerProJahr  # kWh

    return (eMobilität2030 - eMobilitätBisher) * eMobilitätVerbrauch

def faktorRechnung1(verbrauch2022df, wärmepumpeHochrechnung2030, eMobilitätHochrechnung2030):
    gesamtVerbrauch2022 = (otherFactors1(wärmepumpeHochrechnung2030, verbrauch2022df))*1000000000 + 504515946000 # mal1000 weil MWh -> kWh
    return (gesamtVerbrauch2022 + wärmepumpeHochrechnung2030 + eMobilitätHochrechnung2030) / (504515946000) #ges Verbrauch 2021

def prognoseRechnung1(verbrauch2022df, faktor):
    verbrauch2030df = verbrauch2022df['Verbrauch [kWh]'] * faktor
    return verbrauch2030df

def otherFactors1(wärmepumpeHochrechnung2030, verbrauch2022df):
    indHigh = (wärmepumpeHochrechnung2030*(1+3/7))*(72/26)
    indLow = verbrauch2022df['Consumption'].sum()*0.45*0.879/1000000
    indMiddle = 0

    # positive Faktoren
    railway = 5  # TWh
    powerNetLoss = 1
    industry = indHigh

    # negative Faktoren
    efficiency = 51
    other = 6

    return railway  + powerNetLoss - efficiency - other + industry/1000000000

# %%
def process_and_plot_2030_dataMi(production_df, consumption_df, load_profile_df, selected_date):
        
    # Realistisches Ausbau (based on frauenhofer) Szenario 2030 basierend auf 2022 Wetter (mittleres Wetter) ((2021 wäre schlechtes Wetter))
    production_2022df = production_df[production_df['Date'].dt.year == 2022]
    prognoseErzeugung2030_realistic_2022_df = production_2022df.copy()
    prognoseErzeugung2030_realistic_2022_df['Date'] = prognoseErzeugung2030_realistic_2022_df['Date'].map(lambda x: x.replace(year=2030) if not (x.month == 2 and x.day == 29) else x.replace(month=2, day=28, year=2030))

    windonshore_2030_factor_2022_realistic = 1.2921  # 
    windoffshore_2030_factor_2022_realistic = 2.13621  # 
    pv_2030_factor_2022_realistic = 1.821041  # assumig PV will increase by 182%

    def scale_2030_factors(df,windonshore_2030_factor_2022_realistic,windoffshore_2030_factor_2022_realistic,
                                          pv_2030_factor_2022_realistic):
        df_copy = df.copy()
        df_copy['Wind Onshore'] *= windonshore_2030_factor_2022_realistic
        df_copy['Wind Offshore'] *= windoffshore_2030_factor_2022_realistic
        df_copy['Photovoltaic'] *= pv_2030_factor_2022_realistic
        df_copy['Total Production'] = df_copy[['Biomass', 'Hydroelectric', 'Wind Offshore', 'Wind Onshore', 'Photovoltaic', 'Other Renewable']].sum(axis=1)
        return df_copy

    # Scale the data by the factors
    scaled_production_df = scale_2030_factors(prognoseErzeugung2030_realistic_2022_df, windonshore_2030_factor_2022_realistic,windoffshore_2030_factor_2022_realistic,
                                          pv_2030_factor_2022_realistic)

    # Filter the data for the selected date
    scaled_selected_production_df = scaled_production_df[scaled_production_df['Date'] == selected_date]

    verbrauch2030df = energyConsumption2(consumption_df)

    selected_consumption2030df = verbrauch2030df[verbrauch2030df['Date'] == selected_date]
    scaled_selected_production_df = scaled_selected_production_df[scaled_selected_production_df['Date'] == selected_date]
    
    plot_energy_data(selected_consumption2030df, scaled_selected_production_df, selected_date) # Plot the data
    plot_renewable_percentage(scaled_production_df, verbrauch2030df) # Plot the renewable percentage

# Funktion zur Berechnung und Anzeige der aggregierten Daten pro Jahr
# Author: Bjarne, Noah
def energyConsumption2(consumption_df):
    wärmepumpeHochrechnung2030 = wärmepumpe2()
    eMobilitätHochrechnung2030 = eMobilität2()

    verbrauch2022df = consumption_df[consumption_df['Date'].dt.year == 2022]
    prognose2030df = verbrauch2022df.copy()
    faktor = faktorRechnung2(verbrauch2022df, wärmepumpeHochrechnung2030, eMobilitätHochrechnung2030)

    prognose2030df['Date'] = prognose2030df['Date'].map(lambda x: x.replace(year=2030) if not (x.month == 2 and x.day == 29) else x.replace(month=2, day=28, year=2030))

    prognose2030df['Verbrauch [MWh]'] = prognose2030df['Consumption'] * faktor

    combined_df = pd.concat([verbrauch2022df[['Starttime', 'Consumption']], prognose2030df[['Verbrauch [MWh]']]], axis=1)

    return prognose2030df

def wärmepumpe2():
    highScenario = 500000
    lowScenario = 236000
    middleScenario = 368000
    wärmepumpeAnzahl2030 = middleScenario * (2030 - 2023)  # 500k pro Jahr bis 2023

    heizstunden = 2000
    nennleistung = 15  # 15kW
    luftWasserVerhältnis = 206 / 236
    erdwärmeVerhältnis = 30 / 236
    luftWasserJAZ = 3.1
    erdwärmeJAZ = 4.1

    # Berechnung der einzelnen Pumpe
    luftWasserVerbrauch = wärmepumpeVerbrauchImJahr2(heizstunden, nennleistung, luftWasserJAZ)  # in kW/h
    erdwärmeVerbrauch = wärmepumpeVerbrauchImJahr2(heizstunden, nennleistung, erdwärmeJAZ)  # in kW/h

    luftWasserVerhältnisAnzahl = verhältnisAnzahl2(wärmepumpeAnzahl2030, luftWasserVerhältnis)
    erdwärmeVerhältnisAnzahl = verhältnisAnzahl2(wärmepumpeAnzahl2030, erdwärmeVerhältnis)

    return luftWasserVerbrauch * luftWasserVerhältnisAnzahl + erdwärmeVerbrauch * erdwärmeVerhältnisAnzahl  # kWh

# berechnung des Verbrauchs einer Wärmepumpe im Jahr
def wärmepumpeVerbrauchImJahr2(heizstunden, nennleistung, jaz): 
    return (heizstunden * nennleistung) / jaz # (Heizstunden * Nennleistung) / JAZ = Stromverbrauch pro Jahr

def verhältnisAnzahl2(wärmepumpeAnzahl2030, verhältnis):
    return wärmepumpeAnzahl2030 * verhältnis

def eMobilität2():
    highECars = 15000000
    lowECars = 8000000
    middleECars = 11500000

    eMobilität2030 = middleECars  # 15mio bis 20230
    eMobilitätBisher = 1307901  # 1.3 mio
    verbrauchPro100km = 21  # 21kWh
    kilometerProJahr = 15000  # 15.000km

    eMobilitätVerbrauch = (verbrauchPro100km / 100) * kilometerProJahr  # kWh

    return (eMobilität2030 - eMobilitätBisher) * eMobilitätVerbrauch

def faktorRechnung2(verbrauch2022df, wärmepumpeHochrechnung2030, eMobilitätHochrechnung2030):
    gesamtVerbrauch2022 = (otherFactors2(wärmepumpeHochrechnung2030, verbrauch2022df))*1000000000 + 504515946000 # mal1000 weil MWh -> kWh
    return (gesamtVerbrauch2022 + wärmepumpeHochrechnung2030 + eMobilitätHochrechnung2030) / (504515946000) #ges Verbrauch 2021

def prognoseRechnung2(verbrauch2022df, faktor):
    verbrauch2030df = verbrauch2022df['Verbrauch [kWh]'] * faktor
    return verbrauch2030df

def otherFactors2(wärmepumpeHochrechnung2030, verbrauch2022df):
    indHigh = (wärmepumpeHochrechnung2030*(1+3/7))*(72/26)
    indLow = verbrauch2022df['Consumption'].sum()*0.45*0.879/1000000
    indMiddle = 0

    # positive Faktoren
    railway = 5  # TWh
    powerNetLoss = 1
    industry = indMiddle

    # negative Faktoren
    efficiency = 51
    other = 6

    return railway  + powerNetLoss - efficiency - other + industry/1000000000


# %%
def get_date():
    while True:
        selected_date_str = input("Enter the selected date for year 2030 (DD.MM.YYYY): ")
        selected_date = datetime.strptime(selected_date_str, "%d.%m.%Y")
        
        if selected_date.year == 2030:
            return selected_date
        else:
            print("Please enter a date from the year 2030.")

# %%
def main():
    file_production = 'Realisierte_Erzeugung_202001010000_202212312359_Viertelstunde.csv'                   # File names
    file_consumption = 'Realisierter_Stromverbrauch_202001010000_202212312359_Viertelstunde.csv'
    load_profile_df = read_load_profile('Lastprofile_SWKiel.xls')

    # Read and clean data
    production_df, consumption_df, total_renewable_production, total_consumption, data_by_year = read_and_clean_data(file_production, file_consumption)
    
    # Find dark lulls for the years 2020-2022
    #find_dark_lulls_for_years(production_df, columns_to_clean=['Biomass', 'Hydroelectric', 'Wind Offshore', 'Wind Onshore', 'Photovoltaic', 'Other Renewable'])

    # Berechnung und Anzeige des Histogramms für erneuerbare Anteile
    #calculate_and_display_renewable_shares_histogram(total_renewable_production, total_consumption)
    #calculate_and_display_data_by_year(data_by_year)

    #plot_energy_consumption_and_production(production_df, consumption_df, columns_to_clean=['Biomass', 'Hydroelectric', 'Wind Offshore', 'Wind Onshore', 'Photovoltaic', 'Other Renewable'])

    date = get_date() # Get the selected date from the user
    print("gut")
    process_and_plot_2030_dataGut(production_df, consumption_df, load_profile_df, date)
    print("schlecht")
    process_and_plot_2030_dataSchlecht(production_df, consumption_df, load_profile_df, date)
    print("MITTEL")
    process_and_plot_2030_dataMi(production_df, consumption_df, load_profile_df, date)
    
    print(load_profile_df)

if __name__ == "__main__":
    main()


