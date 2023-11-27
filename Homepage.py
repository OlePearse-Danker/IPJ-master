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



st.set_page_config(
    page_title="Homepage"
)
st.sidebar.success("Select a page above.")  

# Apply dark background style
style.use('dark_background')

st.title("WATT-Meister-Consulting Calculator")
st.divider()

# tabs
# --------------------

tab1, tab2 = st.tabs(["As-Is-Analysis", "Prognosis"])

with tab1:
    st.subheader('Energy production and consumption')
    st.write('For the following plots, we collected the electricity market data of Germany for the years 2020, 2021, and 2022 and analyzed the production and consumption. In the first plot, you can see the production and consumption for any specific day in the period from 2020 to 2022.')

    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2022, 12, 31)
    default_date = datetime.date(2020, 1, 1)
    st.write("##")
    input_date = st.date_input(
        "Select a Date",
        value = default_date, 
        min_value=start_date,
        max_value=end_date,
        format="DD.MM.YYYY",
    )
    selected_date = pd.to_datetime(
        input_date,
        format="%d.%m.%Y",
    )



    #Startzeit des Programms
    start_time = time.time()          


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

    # Umrechnungsfaktoren
    M_to_TWh = 1e-6
    MWh_to_GWh = 1e-3

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


    def range1(array1, array2):               # Berechnung der prozentualen Anteile der erneuerbaren Energieerzeugung am Gesamtverbrauch
        if len(array1) != len(array2):
            raise ValueError("Arrays must be the same length")
        
        counts = [0] * 111

        for val1, val2 in zip(array1, array2):
            ratio = val1 / val2
            percent = int(ratio * 100)

            if percent == 100:
                counts[percent] += 1
            elif 0 <= percent < 110:
                counts[percent] += 1

        return counts

    counts = range1(total_renewable_production, total_consumption)
    n = range(111) # Anzahl der Prozenten

    # Ausgabe von Anteilen
    def get_result(array1, array2):
        print("Anteile in %:")
        if len(array1) != len(array2):
            raise ValueError("Arrays must be the same length")
        
        for val1, val2 in zip(array1, array2):
            print( val1, "% :"   , val2)

    get_result(n, counts)
    print("Anzahl der Viertelstunden in 3 Jahren:", sum(counts))
    print()

    # Filtern der Daten für das ausgewählte Datum
    selected_production = production_df[production_df[DATE] == selected_date]
    selected_consumption = consumption_df[consumption_df[DATE] == selected_date]


    end_time = time.time()                         # The time at the end of the program is stored
    duration = end_time - start_time               # Duration of the program is calculated
    print("Duration of the program: ", round(duration, 2))


    # Create a new Plotly subplot figure
    fig_1 = make_subplots()

    fig_1.add_trace(
        go.Scatter(
            x=selected_consumption[STARTTIME].dt.strftime('%H:%M'), 
            y=selected_consumption[CONSUMPTION] * MWh_to_GWh,
            mode='none',
            name='Total Consumption',
            fill='tozeroy',
            fillcolor='#F6C85F'
        )
    )

    # Add the renewable energy production trace
    fig_1.add_trace(
        go.Scatter(
            x=selected_production[STARTTIME].dt.strftime('%H:%M'),
            y=selected_production['Total Production'] * MWh_to_GWh,
            mode='none',
            name='Total Renewable Production',
            fill='tozeroy',
            fillcolor='#0D7F9F'
        )
    )


    fig_1.update_layout(
        title=f'Energy Production and Consumption on {selected_date}',
        xaxis=dict(title='Time (hours)', showgrid=True),
        yaxis=dict(title='Energy (GWh)', showgrid=True, tickformat=".0f"),
        showlegend=True
    )


    # Show the plot using st.plotly_chart
    st.plotly_chart(fig_1)

    #------------------------------------------------

    # Berechnung der prozentualen Anteile der erneuerbaren Energieerzeugung am Gesamtverbrauch

    percent_renewable = total_renewable_production / total_consumption * 100 

    counts, intervals = np.histogram(percent_renewable, bins = np.arange(0, 111, 1))  # Use NumPy to calculate the histogram of the percentage distribution

    x = intervals[:-1]                               # Define the x-axis values as the bin edges
    labels = [f'{i}%' for i in range(0, 111, 1)]     # Create labels for x-axis ticks (von 0 bis 111 in Einzelnschritten)

    fig_2 = go.Figure(data=[go.Bar(x=x, y=counts)])    # Create a bar chart using Plotly

    fig_2.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(0, 111, 5)), ticktext=labels[::5]))  # X-axis label settings

    # Title and axis labels settings
    fig_2.update_layout(title='Number of fifteen-minute intervals in years 2020-2022 with 0-110% share of renewable energy',
                    xaxis_title='Percentage of renewable energy',
                    yaxis_title='Number of fifteen-minute intervals')


    st.plotly_chart(fig_2)                            # Show the plot using st.plotly_chart

    # ------------------------------------------------
    # displaying yearly production of renewable energy

    # df_c.set_index('Datum', inplace=True)

    st.subheader('Production of Renewable Energy')
    st.write('As renewable considered are: waterpower, biomass, wind onshore/offshore, photovoltaics and other renewables')


    year_options = [2020, 2021, 2022]
    year = st.selectbox('Which year would you like to see?', year_options)

    # Filter the production_df dataframe for the selected year
    filtered_production_df = production_df[production_df[DATE].dt.year == year]
    filtered_consumption_df = consumption_df[consumption_df[DATE].dt.year == year]

    # Compute the total production for each renewable energy type
    total_biomass = filtered_production_df[BIOMAS].sum()
    total_waterpower = filtered_production_df[HYDROELECTRIC].sum()
    total_windoff = filtered_production_df[WIND_OFFSHORE].sum()
    total_windon = filtered_production_df[WIND_ONSHORE].sum()
    total_pv = filtered_production_df[PHOTOVOLTAIC].sum()
    total_other_ree = filtered_production_df[OTHER_RENEWABLE].sum()

    # Calculate the sum of all the total values for the selected year
    total_ree_sum = total_biomass + total_waterpower + total_windoff + total_windon + total_pv + total_other_ree
    total_consumption_y = filtered_consumption_df[CONSUMPTION].sum()

    # Create a bar chart to display the yearly production of renewable energy
    fig_3 = px.bar(x=[BIOMAS, HYDROELECTRIC, WIND_OFFSHORE, WIND_ONSHORE, PHOTOVOLTAIC, OTHER_RENEWABLE],
                y=[total_biomass * MWh_to_GWh, total_waterpower * MWh_to_GWh, total_windoff * MWh_to_GWh, total_windon * MWh_to_GWh, total_pv * MWh_to_GWh, total_other_ree * MWh_to_GWh],
                labels={'x': 'Renewable Energy Type', 'y': 'Total Production (GWh)'},
                title=f"Yearly Production of Renewable Energy - {year}")

    fig_3.data[0].x = ['biomass', 'hydroelectric', 'wind offshore', 'wind onshore', 'photovoltaic', 'other_renewable']

    # Display the bar chart using st.plotly_chart
    st.plotly_chart(fig_3)

    st.markdown('#### Total Renewable Energy Production and Total Consumption in TWh')
    st.markdown('For the years 2021 to 2022 the values are compared to the previous years')

    if year == 2021 or year == 2022:

        filtered_production_df_prev = production_df[production_df[DATE].dt.year == year-1]
        total_biomass_prev = filtered_production_df_prev[BIOMAS].sum()
        total_waterpower_prev = filtered_production_df_prev[HYDROELECTRIC].sum()
        total_windoff_prev = filtered_production_df_prev[WIND_OFFSHORE].sum()
        total_windon_prev = filtered_production_df_prev[WIND_ONSHORE].sum()
        total_pv_prev = filtered_production_df_prev[PHOTOVOLTAIC].sum()
        total_other_ree_prev = filtered_production_df_prev[OTHER_RENEWABLE].sum()

        total_ree_sum_prev = total_biomass_prev + total_waterpower_prev + total_windoff_prev + total_windon_prev + total_pv_prev + total_other_ree_prev

        diff_production = f'{(total_ree_sum - total_ree_sum_prev) * M_to_TWh:.2f}' + ' TWh'

        total_consumption_y_prev = consumption_df[consumption_df[DATE].dt.year == year-1][CONSUMPTION].sum()
        diff_consumption = f'{(total_consumption_y - total_consumption_y_prev) * M_to_TWh:.2f}' + ' TWh'
    else:
        diff_production = 0
        diff_consumption = 0


    col1, col2 = st.columns(2)

    with col1:
        st.metric(label='Total Renewable Energy Production (TWh)', value=f'{total_ree_sum * M_to_TWh:.2f}', delta=diff_production, delta_color='normal')

    with col2:
        st.metric(label='Total Consumption (TWh)', value=f'{total_consumption_y * M_to_TWh:.2f}', delta=diff_consumption, delta_color='normal')



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
        start_date = datetime.date(2020, 1, 1)
        end_date = datetime.date(2022, 12, 31)

        dark_lulls_dict = {"up to 10%": [], "up to 20%": []}
        current_date = pd.Timestamp(start_date)
        
        while current_date <= pd.Timestamp(end_date):
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

        amount_10 = len(dark_lulls_dict["up to 10%"])
        amount_20 = len(dark_lulls_dict["up to 20%"])

        return amount_10, amount_20


    st.markdown('#### Amount of dark lulls in from 2020 to 2022')
    st.markdown('The following metrics shows the amount of dark lulls for the years 2020 to 2022. A dark lull is defined as a day where the renewable energy production is less than 10% or 20% of the installed power.')


    amount_10, amount_20 = find_dark_lulls_for_years(production_df, installed_power_dict)


    col1, col2 = st.columns(2)

    with col1:
        st.metric(label='Number of days up to 10%', value=amount_10)

    with col2:
        st.metric(label='Number of days up to 20%', value=amount_20)



# ------------------------------------------------

# first forecast by Lucas

with tab2:

    # Define the factors
    windonshore_2030_factor = 2.03563  # assuming Wind Onshore will increase by 203%
    windoffshore_2030_factor = 3.76979  # assuming Wind Offshore will 376% increase
    pv_2030_factor = 3.5593  # assuming PV will increase by 350%

    def scale_2030_factors(df, windonshore_factor, windoffshore_factor, pv_factor):
        df_copy = df.copy()
        df_copy[WIND_ONSHORE] *= windonshore_factor
        df_copy[WIND_OFFSHORE] *= windoffshore_factor
        df_copy[PHOTOVOLTAIC] *= pv_factor
        df_copy['Total Production'] = df_copy[columns_to_clean].sum(axis=1)
        return df_copy


    def show_figure_4():

        # Scale the data by the factors
        scaled_production_df = scale_2030_factors(production_df, windonshore_2030_factor, windoffshore_2030_factor, pv_2030_factor)

        # Filter the data for the selected date
        scaled_selected_production = scaled_production_df[scaled_production_df[DATE] == selected_date]

        #code to do 2030 quarter hours
        total_scaled_renewable_production = scaled_production_df[columns_to_clean].sum(axis=1)

        # Berechnung der prozentualen Anteile der erneuerbaren Energieerzeugung am Gesamtverbrauch
        percent_renewable = total_scaled_renewable_production / total_consumption * 100 

        counts, intervals = np.histogram(percent_renewable, bins = np.arange(0, 330, 1))  # Use NumPy to calculate the histogram of the percentage distribution

        x = intervals[:-1]          # Define the x-axis values as the bin edges
        labels = [f'{i}%' for i in range(0, 330, 1)] # Create labels for x-axis ticks (von 0 bis 111 in Einzelnschritten)

        fig_4 = go.Figure(data=[go.Bar(x=x, y=counts)])    # Create a bar chart using Plotly
        fig_4.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(0, 330, 5)), ticktext=labels[::5]))  # X-axis label settings

        # Title and axis labels settings
        fig_4.update_layout(title='Number of quarters in years 2030 - 2032 with 0-330% share of renewable energy',
                        xaxis_title='Percentage of renewable energy production',
                        yaxis_title='Number of quarters')

        st.plotly_chart(fig_4)


    st.subheader('Forecast of Renewable Energy production')

    show_figure_4()


# ------------------------------------------------
# Tryout for the 2030 production forecast from Noah

#--------------------------------------------------------------------------
# Abfrage datum




    # Funktion
    def energyConsumption(consumption_df):
        wärmepumpeHochrechnung2030 = wärmepumpe()
        eMobilitätHochrechnung2030 = eMobilität()

        print('\n', 'wärmepumpeHochrechnung2030', f"{wärmepumpeHochrechnung2030:,.0f}".replace(",", "."))
        print('\n', 'eMobilitätHochrechnung2030', f"{eMobilitätHochrechnung2030:,.0f}".replace(",", "."))

        verbrauch2022df = consumption_df[consumption_df['Datum'].dt.year == 2022]
        prognose2030df = verbrauch2022df.copy()
        faktor = faktorRechnung(verbrauch2022df, wärmepumpeHochrechnung2030, eMobilitätHochrechnung2030)
        print(faktor)
        # Change the year in 'Datum' column to 2030
        prognose2030df['Datum'] = prognose2030df['Datum'].map(lambda x: x.replace(year=2030))

        prognose2030df['Verbrauch [MWh]'] = prognose2030df['Gesamt (Netzlast) [MWh] Originalauflösungen'] * faktor

        combined_df = pd.concat([verbrauch2022df[['Anfang', 'Gesamt (Netzlast) [MWh] Originalauflösungen']], prognose2030df[['Verbrauch [MWh]']]], axis=1)
        print(combined_df[['Gesamt (Netzlast) [MWh] Originalauflösungen', 'Verbrauch [MWh]']])

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
        gesamtVerbrauch2022 = otherFactors() + verbrauch2022df[
            'Gesamt (Netzlast) [MWh] Originalauflösungen'].sum() * 1000  # mal1000 weil MWh -> kWh
        print('\n', 'gesamtVerbrauch2022', f"{gesamtVerbrauch2022:,.0f}".replace(",", "."))
        return (gesamtVerbrauch2022 + wärmepumpeHochrechnung2030 + eMobilitätHochrechnung2030) / gesamtVerbrauch2022


    def prognoseRechnung(verbrauch2022df, faktor):
        verbrauch2030df = verbrauch2022df['Verbrauch [kWh]'] * faktor
        return verbrauch2030df


    def otherFactors():
        # positive Faktoren
        railway = 5000  # kWh
        batterieProdAndServerRooms = 13000  # kwh
        powerNetLoss = 1000

        # negative Faktoren
        efficiency = 51000
        other = 6000
        return railway + batterieProdAndServerRooms + powerNetLoss - efficiency - other


    start_date = datetime.date(2030, 1, 1)
    end_date = datetime.date(2030, 12, 31)
    default_date = datetime.date(2030, 1, 1)
    st.write("##")
    input_date = st.date_input(
        "Select a Date",
        value = default_date, 
        min_value=start_date,
        max_value=end_date,
        format="DD.MM.YYYY",
    )
    selected_date = pd.to_datetime(
        input_date,
        format="%d.%m.%Y",
    )

    # Define a dataframe of the production of 2022
    production_2022df = production_df[production_df[DATE].dt.year == 2022]
    prognoseErzeugung2030df = production_2022df.copy()
    prognoseErzeugung2030df['Datum'] = prognoseErzeugung2030df['Datum'].map(lambda x: x.replace(year=2030))

    # Define a dataframe of the consumption of 2022
    consumption_2022df = consumption_df[consumption_df[DATE].dt.year == 2022]
    prognoseVerbrauch2030df = consumption_2022df.copy()
    prognoseVerbrauch2030df['Datum'] = prognoseVerbrauch2030df['Datum'].map(lambda x: x.replace(year=2030))


    def scale_2030_factorsConsumption(df, Verbrauch2022_2030_factor):
        df_copy = df.copy()
        df_copy[CONSUMPTION] *= Verbrauch2022_2030_factor
        return df_copy


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

    fig_5 = make_subplots()

    # Add the energy consumption trace
    fig_5.add_trace(
        go.Scatter(
            x=selected_consumption2030df[STARTTIME].dt.strftime('%H:%M'),
            y=selected_consumption2030df['Verbrauch [MWh]'],
            mode='lines',
            name='Total Consumption',
            fill='tozeroy'
        )
    )

    # Add the renewable energy production trace
    fig_5.add_trace(
        go.Scatter(
            x=scaled_selected_production_df[STARTTIME].dt.strftime('%H:%M'),
            y=scaled_selected_production_df['Total Production'],
            mode='lines',
            name='Total Renewable Production',
            fill='tozeroy'
        )
    )

    fig_5.update_layout(
        title=f'Energy Production and Consumption on {selected_date}',
        xaxis=dict(title='Time (hours)'),
        yaxis=dict(title='Energy (MWh)'),
        showlegend=True
    )

    st.plotly_chart(fig_5)

    # code to do 2030 quarter hours
    total_scaled_renewable_production = scaled_production_df[columns_to_clean].sum(axis=1)
    total_consumption = verbrauch2030df['Verbrauch [MWh]']

    # Berechnung der prozentualen Anteile der erneuerbaren Energieerzeugung am Gesamtverbrauch
    percent_renewable = total_scaled_renewable_production / total_consumption * 100

    counts, intervals = np.histogram(percent_renewable, bins=np.arange(0, 330, 1))  # Use NumPy to calculate the histogram of the percentage distribution

    x = intervals[:-1]  # Define the x-axis values as the bin edges
    labels = [f'{i}%' for i in range(0, 330, 1)]  # Create labels for x-axis ticks (von 0 bis 111 in Einzelnschritten)

    fig_6 = go.Figure(data=[go.Bar(x=x, y=counts)])  # Create a bar chart using Plotly
    fig_6.update_layout(
        xaxis=dict(tickmode='array', tickvals=list(range(0, 330, 5)), ticktext=labels[::5]))  # X-axis label settings

    # Title and axis labels settings
    fig_6.update_layout(title='Anzahl der Viertelstunden im Jahren 2030 mit 0-330 % EE-Anteil',
                    xaxis_title='Prozentsatz erneuerbarer Energie',
                    yaxis_title='Anzahl der Viertelstunden')

    st.plotly_chart(fig_6)

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

    fig_7 = go.Figure()

    # Fügen Sie einen Balken für die Anzahl der Viertelstunden für jeden Prozentsatz hinzu
    fig_7.add_trace(go.Bar(x=result_df['Prozentsatz'], y=result_df['Anzahl_Viertelstunden']))

    # Aktualisieren Sie das Layout für Titel und Achsenbeschriftungen
    fig_7.update_layout(
        title='Anzahl der Viertelstunden mit erneuerbarer Energieerzeugung über oder gleich dem Verbrauch',
        xaxis=dict(title='Prozentsatz erneuerbarer Energie'),
        yaxis=dict(title='Anzahl der Viertelstunden')
    )

    # Zeigen Sie den Plot an
    st.plotly_chart(fig_7)
