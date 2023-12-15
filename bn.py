import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import datetime
import time
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Function to process and plot data for the year 2030
def process_and_plot_2030_data(production_df, consumption_df):
    # Abfrage datum
    while True:
        selected_date_str = input("Bitte geben Sie ein Datum IM JAHR 2030 im Format TT.MM.JJJJ ein: ")
        selected_date = datetime.strptime(selected_date_str, "%d.%m.%Y")

        if selected_date.year == 2030:
            break
        else:
            print("Bitte geben Sie ein Datum aus dem Jahr 2030 an.")

    # Define a dataframe of the production of 2022
    production_2022df = production_df[production_df['Date'].dt.year == 2022]
    prognoseErzeugung2030df = production_2022df.copy()
    prognoseErzeugung2030df['Date'] = prognoseErzeugung2030df['Date'].map(lambda x: x.replace(year=2030))

    # Define a dataframe of the consumption of 2022
    consumption_2022df = consumption_df[consumption_df['Date'].dt.year == 2022]
    prognoseVerbrauch2030df = consumption_2022df.copy()
    prognoseVerbrauch2030df['Date'] = prognoseVerbrauch2030df['Date'].map(lambda x: x.replace(year=2030))

    # CHANGE LATER Define the factors Verbrauch 2022 to 2030 von Bjarne Noah
    # Verbrauch2022_2030_factor = 1.157  # assuming consumption will increase by 10% from 2022 compared to 2030

    def scale_2030_factorsConsumption(df, Verbrauch2022_2030_factor):
        df_copy = df.copy()
        df_copy['Consumption'] *= Verbrauch2022_2030_factor
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
        df_copy['Wind Onshore'] *= windonshore_factor
        df_copy['Wind Offshore'] *= windoffshore_factor
        df_copy['Photovoltaic'] *= pv_factor
        df_copy['Total Production'] = df_copy[['Biomass', 'Hydroelectric', 'Wind Offshore', 'Wind Onshore', 'Photovoltaic', 'Other Renewable']].sum(axis=1)
        return df_copy

    # Scale the data by the factors
    scaled_production_df = scale_2030_factors(prognoseErzeugung2030df, windonshore_2030_factor, windoffshore_2030_factor,
                                              pv_2030_factor)

    # Filter the data for the selected date
    scaled_selected_production_df = scaled_production_df[scaled_production_df['Date'] == selected_date]

    verbrauch2030df = energyConsumption(consumption_df)

    # scaled_consumption_df = scale_2030_factorsConsumption(prognoseVerbrauch2030df, Verbrauch2022_2030_factor)

    # Filter the data for the selected date
    # scaled_selected_consumption_df = scaled_consumption_df[scaled_consumption_df[DATE] == selected_date]

    ### subplot only consumption 2030

    ## Plotly daily production and consumption 2030

    selected_consumption2030df = verbrauch2030df[verbrauch2030df['Date'] == selected_date]
    scaled_selected_production_df = scaled_selected_production_df[scaled_selected_production_df['Date'] == selected_date]

    fig = make_subplots()

    # Add the energy consumption trace
    fig.add_trace(
        go.Scatter(
            x=selected_consumption2030df['Starttime'].dt.strftime('%H:%M'),
            y=selected_consumption2030df['Verbrauch [MWh]'],
            mode='lines',
            name='Total Consumption',
            fill='tozeroy'
        )
    )

    # Add the renewable energy production trace
    fig.add_trace(
        go.Scatter(
            x=scaled_selected_production_df['Starttime'].dt.strftime('%H:%M'),
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
        # Zählen die Viertelstunden über oder gleich dem Prozentsatz
        anzahlViertelstundenProzent = len(percent_renewable[percent_renewable >= i])
         # Fügen Sie einen Datensatz zum Speichern in die Liste hinzu
        data.append({'Prozentsatz': i, 'Anzahl_Viertelstunden': anzahlViertelstundenProzent})
    
    result_df = pd.DataFrame(data) # DataFrame erstellen

    #print(result_df) # Check the result

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




# Funktion zur Berechnung und Anzeige der aggregierten Daten pro Jahr
# Author: Bjarne, Noah
# Funktion
def energyConsumption(consumption_df):
    wärmepumpeHochrechnung2030 = wärmepumpe()
    eMobilitätHochrechnung2030 = eMobilität()

    print('\n', 'wärmepumpeHochrechnung2030', f"{wärmepumpeHochrechnung2030:,.0f}".replace(",", "."))
    print('\n', 'eMobilitätHochrechnung2030', f"{eMobilitätHochrechnung2030:,.0f}".replace(",", "."))

    verbrauch2022df = consumption_df[consumption_df['Datum'].dt.year == 2021]
    prognose2030df = verbrauch2022df.copy()
    faktor = faktorRechnung(verbrauch2022df, wärmepumpeHochrechnung2030, eMobilitätHochrechnung2030)
    print(faktor)
    # Change the year in 'Datum' column to 2030
    prognose2030df['Datum'] = prognose2030df['Datum'].map(lambda x: x.replace(year=2030))

    prognose2030df['Verbrauch [MWh]'] = prognose2030df['Consumption'] * faktor

    combined_df = pd.concat([verbrauch2022df[['Starttime', 'Consumption']], prognose2030df[['Verbrauch [MWh]']]], axis=1)
    #print(combined_df[['Gesamt (Netzlast) [MWh] Originalauflösungen', 'Verbrauch [MWh]']])

    print("Verbrauch 2030:", prognose2030df['Verbrauch [MWh]'].sum()/1000 , "TWhhusp\n")
    print("Consumption 2022:", prognose2030df['Consumption'].sum()/1000 , "TWh\n")

    return prognose2030df

def wärmepumpe():
    wärmepumpeAnzahl2030 = 500000 * (2030 - 2023)  # 500k pro Jahr bis 2023

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
    # (Heizstunden * Nennleistung) / JAZ = Stromverbrauch pro Jahr
    return (heizstunden * nennleistung) / jaz

def verhältnisAnzahl(wärmepumpeAnzahl2030, verhältnis):
    return wärmepumpeAnzahl2030 * verhältnis

def eMobilität():
    eMobilität2030 = 15000000  # 15mio bis 20230
    eMobilitätBisher = 1307901  # 1.3 mio
    verbrauchPro100km = 21  # 21kWh
    kilometerProJahr = 15000  # 15.000km

    eMobilitätVerbrauch = (verbrauchPro100km / 100) * kilometerProJahr  # kWh

    return (eMobilität2030 - eMobilitätBisher) * eMobilitätVerbrauch

def faktorRechnung(verbrauch2022df, wärmepumpeHochrechnung2030, eMobilitätHochrechnung2030):
    gesamtVerbrauch2022 = (otherFactors(wärmepumpeHochrechnung2030))*1000000000 + verbrauch2022df['Consumption'].sum() * 1000  # mal1000 weil MWh -> kWh
    #print('\n', 'gesamtVerbrauch2022', f"{gesamtVerbrauch2022:,.0f}".replace(",", "."))
    return (gesamtVerbrauch2022 + wärmepumpeHochrechnung2030 + eMobilitätHochrechnung2030) / (verbrauch2022df['Consumption'].sum()*1000)

def prognoseRechnung(verbrauch2022df, faktor):
    verbrauch2030df = verbrauch2022df['Verbrauch [kWh]'] * faktor
    return verbrauch2030df

def otherFactors(wärmepumpeHochrechnung2030):
    # positive Faktoren
    railway = 5  # TWh
    batterieProdAndServerRooms = 13  # Twh
    powerNetLoss = 1
    industry = 126 #bei 32TWhhh
    industry2 = (wärmepumpeHochrechnung2030*(1+3/7))*(72/26)

    # negative Faktoren
    efficiency = 51
    other = 6

    return railway + batterieProdAndServerRooms + powerNetLoss - efficiency - other + industry2/1000000000