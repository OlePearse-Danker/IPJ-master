import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import datetime
import time
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import xlrd
from plotly.subplots import make_subplots



st.set_page_config(
    page_title="Homepage",
    layout='wide'
)
st.sidebar.success("Select a page above.")  

# Apply dark background style
style.use('dark_background')

st.title("WATT-Meister-Consulting Calculator")
st.divider()

# tabs
# --------------------

tab1, tab2, tab3, tab4 = st.tabs(["As-Is-Analysis", "Preset Scenarios", "Load Profile Scenario", "Scenario Builder"])

with tab1:
    st.header('Energy production and consumption')
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
        title=f'Energy Production and Consumption on {selected_date.strftime("%d.%m.%Y")}',
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
            # print(f"No installed power found for the year {year}.")
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

    def find_dark_lulls_for_years(production_df, installed_power_dict,year):
        # Loop through all days in the years 2020 to 2022
        start_date = datetime.date(year, 1, 1)
        end_date = datetime.date(year, 12, 31)

        dark_lulls_dict = {"up to 10%": [], "up to 20%": []}
        current_date = pd.Timestamp(start_date)

        amount_10 = 0
        amount_20 = 0
        
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


    year = selected_date.year

    def calculate_metrics(year):
        amount_10, amount_20 = find_dark_lulls_for_years(production_df, installed_power_dict, year)
        return amount_10, amount_20

    def update_metrics(year):

        previous_year = year - 1  # Calculate the previous year

        amount_10, amount_20 = calculate_metrics(year)

        if year == 2021 or 2022:
            amount_10_previous, amount_20_previous = calculate_metrics(previous_year)
        else:
            amount_10_previous, amount_20_previous = 0, 0

        delta_10 = amount_10 - amount_10_previous
        delta_20 = amount_20 - amount_20_previous

        col1.metric("Number of days up to 10%", amount_10, delta=delta_10, delta_color='normal')
        col2.metric("Number of days up to 20%", amount_20, delta=delta_20, delta_color='normal')
        

    year_options = [2020, 2021, 2022]

    st.markdown(f'#### Amount of dark lulls in {year}')
    st.markdown('The following metrics show the amount of dark lulls for the years 2020 to 2022. A dark lull is defined as a day where the renewable energy production is less than 10% or 20% of the installed power.')
    st.markdown('For the years 2021 and 2022 the amount of dark lulls is compared to the previous year.')

    year = st.selectbox('Which year would you like to see?', year_options, key='year_selectbox')

    col1, col2 = st.columns(2)

    update_metrics(year)




# ------------------------------------------------

# first forecast by Lucas

with tab2:

    # # Define the factors
    # windonshore_2030_factor = 2.03563  # assuming Wind Onshore will increase by 203%
    # windoffshore_2030_factor = 3.76979  # assuming Wind Offshore will 376% increase
    # pv_2030_factor = 3.5593  # assuming PV will increase by 350%

    # def scale_2030_factors(df, windonshore_factor, windoffshore_factor, pv_factor):
    #     df_copy = df.copy()
    #     df_copy[WIND_ONSHORE] *= windonshore_factor
    #     df_copy[WIND_OFFSHORE] *= windoffshore_factor
    #     df_copy[PHOTOVOLTAIC] *= pv_factor
    #     df_copy['Total Production'] = df_copy[columns_to_clean].sum(axis=1)
    #     return df_copy


    # def show_figure_4():

    #     # Scale the data by the factors
    #     scaled_production_df = scale_2030_factors(production_df, windonshore_2030_factor, windoffshore_2030_factor, pv_2030_factor)

    #     # Filter the data for the selected date
    #     scaled_selected_production = scaled_production_df[scaled_production_df[DATE] == selected_date]

    #     #code to do 2030 quarter hours
    #     total_scaled_renewable_production = scaled_production_df[columns_to_clean].sum(axis=1)

    #     # Berechnung der prozentualen Anteile der erneuerbaren Energieerzeugung am Gesamtverbrauch
    #     percent_renewable = total_scaled_renewable_production / total_consumption * 100 

    #     counts, intervals = np.histogram(percent_renewable, bins = np.arange(0, 330, 1))  # Use NumPy to calculate the histogram of the percentage distribution

    #     x = intervals[:-1]          # Define the x-axis values as the bin edges
    #     labels = [f'{i}%' for i in range(0, 330, 1)] # Create labels for x-axis ticks (von 0 bis 111 in Einzelnschritten)

    #     fig_4 = go.Figure(data=[go.Bar(x=x, y=counts)])    # Create a bar chart using Plotly
    #     fig_4.update_layout(xaxis=dict(tickmode='array', tickvals=list(range(0, 330, 5)), ticktext=labels[::5]))  # X-axis label settings

    #     # Title and axis labels settings
    #     fig_4.update_layout(title='Number of quarters in years 2030 - 2032 with 0-330% share of renewable energy',
    #                     xaxis_title='Percentage of renewable energy production',
    #                     yaxis_title='Number of quarters')

    #     st.plotly_chart(fig_4)


    # st.subheader('Forecast of Renewable Energy production')

    # show_figure_4()


#--------------------------------------------------------------------------
# GOOD SCENARIO METHODS BEGIN
#--------------------------------------------------------------------------

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
        load_profile_df[cols_to_update] = load_profile_df[cols_to_update].applymap(lambda x: x * 32 * 10**3)
        
        return load_profile_df

    def consumption_with_load_profile(selected_consumption2030df, load_profile_df, selected_date):
        # Convert selected_date to a datetime object
        selected_date = pd.to_datetime(selected_date)

        # Determine the season and day of the week
        if selected_date.month >= 10 and selected_date.day >= 15 or selected_date.month <= 3 and selected_date.day <= 15:
            season = 'Winter'
        else:
            season = 'Summer'

        day_of_week = selected_date.day_name()

        # Map the day of the week to the corresponding column in load_profile_df
        if day_of_week == 'Sunday':
            day_column = 'Sunday_' + season
        elif day_of_week == 'Saturday':
            day_column = 'Saturday_' + season
        else:
            day_column = 'Weekday_' + season

        # Ensure both dataframes have the same index
        load_profile_df = load_profile_df.set_index(selected_consumption2030df.index)

        # Merge selected_consumption2030df with the correct column of load_profile_df
        merged_df = selected_consumption2030df.merge(load_profile_df[[day_column]], left_index=True, right_index=True)

        # Add the values in 'Verbrauch [MWh]' with the values in the day_column
        merged_df['Verbrauch [MWh]'] = merged_df['Verbrauch [MWh]'] + merged_df[day_column]

        # Drop the unnecessary columns
        merged_df.drop(columns=[day_column], inplace=True)

        return merged_df

    def scale_2030_factors(df,windonshore_2030_factor,windoffshore_2030_factor, pv_2030_factor):
        df_copy = df.copy()
        df_copy[WIND_ONSHORE] *= windonshore_2030_factor
        df_copy[WIND_OFFSHORE] *= windoffshore_2030_factor
        df_copy[PHOTOVOLTAIC] *= pv_2030_factor
        df_copy['Total Production'] = df_copy[[BIOMAS, HYDROELECTRIC, WIND_OFFSHORE, WIND_ONSHORE, PHOTOVOLTAIC, OTHER_RENEWABLE]].sum(axis=1)
        return df_copy
    

    def plot_renewable_percentage(scaled_production_df, verbrauch2030df):
        total_scaled_renewable_production = scaled_production_df[[BIOMAS, HYDROELECTRIC, WIND_OFFSHORE, WIND_ONSHORE, PHOTOVOLTAIC, OTHER_RENEWABLE]].sum(axis=1)
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
        fig.update_layout(title='Number of quarter hours in 2030 with 0-330% share of renewable energy',
                    xaxis_title='Percentage of renewable energy',
                    yaxis_title='Number of quarter hours')

        st.plotly_chart(fig)


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
            title='Number of quarters with renewable energy generation equal to or greater than consumption in 2030',
            xaxis=dict(title='Percentage of renewable energy'),
            yaxis=dict(title='Number of quarter hours')
        )

        st.plotly_chart(fig)


    def plot_energy_data(consumption_df, production_df, selected_date):
        fig = make_subplots()

        # Add the energy consumption trace
        fig.add_trace(
            go.Scatter(
                x=consumption_df[STARTTIME].dt.strftime('%H:%M'),
                y=consumption_df['Verbrauch [MWh]'],
                mode='lines',
                name='Total Consumption',
                fill='tozeroy'
            )
        )

        # Add the renewable energy production trace
        fig.add_trace(
            go.Scatter(
                x=production_df[STARTTIME].dt.strftime('%H:%M'),
                y=production_df['Total Production'],
                mode='lines',
                name='Total Renewable Production',
                fill='tozeroy'
            )
        )

        fig.update_layout(
            title=f'Energy Production and Consumption on {selected_date.strftime("%d.%m.%Y")}',
            xaxis=dict(title='Time (hours)'),
            yaxis=dict(title='Energy (MWh)'),
            showlegend=True
        )

        st.plotly_chart(fig)

    def process_and_plot_2030_dataGut(production_df, consumption_df, load_profile_df, selected_date):
        
        # POSITIVE SCENARIO Production based on 2020 and BMWK goals
        production_2020df = production_df[production_df[DATE].dt.year == 2020]
        prognoseErzeugung2030_positive_df = production_2020df.copy()
        #prognoseErzeugung2030_positive_df['Date'] = prognoseErzeugung2030_positive_df['Date'].map(lambda x: x.replace(year=2030))
        prognoseErzeugung2030_positive_df[DATE] = prognoseErzeugung2030_positive_df[DATE].map(
        lambda x: x.replace(year=2030) if not (x.month == 2 and x.day == 29) else x.replace(month=2, day=28, year=2030))

        windonshore_2030_factor_2020_positive = 2.13589  # 
        windoffshore_2030_factor_2020_postive = 3.92721  #
        pv_2030_factor_2020_postive = 4.2361193  # assumig PV will increase by 423%

        # Scale the data by the factors
        scaled_production_df = scale_2030_factors(prognoseErzeugung2030_positive_df, windonshore_2030_factor_2020_positive,windoffshore_2030_factor_2020_postive,
                                            pv_2030_factor_2020_postive)

        # Filter the data for the selected date
        scaled_selected_production_df = scaled_production_df[scaled_production_df[DATE] == selected_date]

        verbrauch2030df = energyConsumption(consumption_df)

        selected_consumption2030df = verbrauch2030df[verbrauch2030df[DATE] == selected_date]
        scaled_selected_production_df = scaled_selected_production_df[scaled_selected_production_df[DATE] == selected_date]

        plot_energy_data(selected_consumption2030df, scaled_selected_production_df, selected_date)
        plot_renewable_percentage(scaled_production_df, verbrauch2030df)

        #------------------------------------
        # TEST with yearly metrics

        year = selected_date.year
        
        # Filter the production_df dataframe for the selected year
        filtered_production_df = scaled_production_df[scaled_production_df[DATE].dt.year == year]
        filtered_consumption_df = verbrauch2030df[verbrauch2030df[DATE].dt.year == year]

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

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label='Total Renewable Energy Production (TWh)', value=f'{total_ree_sum * M_to_TWh:.2f}')
        

        with col2:
            # st.metric(label='Total Consumption (TWh)', value=f'{total_consumption_y * M_to_TWh:.2f}')
            st.metric(label='Total Consumption (TWh)', value=513.6)

        #------------------------------------

        return scaled_production_df, verbrauch2030df

    # Funktion zur Berechnung und Anzeige der aggregierten Daten pro Jahr
    # Author: Bjarne, Noah
    def energyConsumption(consumption_df):
        wärmepumpeHochrechnung2030 = wärmepumpe()
        eMobilitätHochrechnung2030 = eMobilität()

        verbrauch2022df = consumption_df[consumption_df[DATE].dt.year == 2020]
        prognose2030df = verbrauch2022df.copy()
        faktor = faktorRechnung(verbrauch2022df, wärmepumpeHochrechnung2030, eMobilitätHochrechnung2030)
        print("Verbr df:", prognose2030df)
        print("Faktor: ", faktor)
        # Change the year in 'Datum' column to 2030
        prognose2030df[DATE] = prognose2030df[DATE].map(lambda x: x.replace(year=2030) if not (x.month == 2 and x.day == 29) else x.replace(month=2, day=28, year=2030))

        prognose2030df['Verbrauch [MWh]'] = prognose2030df[CONSUMPTION] * faktor

        combined_df = pd.concat([verbrauch2022df[[STARTTIME, CONSUMPTION]], prognose2030df[['Verbrauch [MWh]']]], axis=1)
        print("Verbrauch 2030:", prognose2030df['Verbrauch [MWh]'].sum()/1000 , "TWhhusp\n")
        print("Consumption 2022:", prognose2030df[CONSUMPTION].sum()/1000 , "TWh\n")

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
        indLow = -verbrauch2022df[CONSUMPTION].sum()*1000*0.45*0.121
        indMiddle = 0

        # positive Faktoren
        railway = 5  # TWh
        powerNetLoss = 1
        industry = indLow

        # negative Faktoren
        efficiency = 51
        other = 6

        return railway  + powerNetLoss - other + industry/1000000000

#--------------------------------------------------------------------------
# GOOD SCENARIO METHODS END
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# GOOD SCENARIO WITH LOAD PROFILE METHODS BEGIN
#--------------------------------------------------------------------------

    # Function to process and plot data for the year 2030
    def process_and_plot_2030_dataGut2(production_df, consumption_df, load_profile_df, selected_date):
        # Define constants
        windonshore_2030_factor_2020_positive = 2.13589
        windoffshore_2030_factor_2020_postive = 3.92721
        pv_2030_factor_2020_postive = 4.2361193

        # Process production data
        production_2020df = production_df[production_df[DATE].dt.year == 2020]
        prognoseErzeugung2030_positive_df = production_2020df.copy()
        prognoseErzeugung2030_positive_df[DATE] = prognoseErzeugung2030_positive_df[DATE].map(
            lambda x: x.replace(year=2030) if not (x.month == 2 and x.day == 29) else x.replace(month=2, day=28, year=2030))
        scaled_production_df = scale_2030_factors(prognoseErzeugung2030_positive_df, windonshore_2030_factor_2020_positive,windoffshore_2030_factor_2020_postive, pv_2030_factor_2020_postive)

        # Process consumption data
        verbrauch2030df = energyConsumption5(consumption_df)
        selected_consumption2030df = verbrauch2030df[verbrauch2030df[DATE] == selected_date]

        # Filter production data for the selected date
        scaled_selected_production_df = scaled_production_df[scaled_production_df[DATE] == selected_date]

        print("Bevor ")
        print(selected_consumption2030df['Verbrauch [MWh]'])
        a = consumption_with_load_profile(selected_consumption2030df, load_profile_df, selected_date)
        print("Nach ")
        print(a['Verbrauch [MWh]'])

        # Plot data
        plot_energy_data(a, scaled_selected_production_df, selected_date)
        plot_renewable_percentage(scaled_production_df, verbrauch2030df)   
    # Funktion zur Berechnung und Anzeige der aggregierten Daten pro Jahr

    def energyConsumption5(consumption_df):
        verbrauch2022df = consumption_df[consumption_df[DATE].dt.year == 2020]
        prognose2030df = verbrauch2022df.copy()
        faktor = faktorRechnung5(verbrauch2022df, eMobilität5())
        print("Faktor: ", faktor)
        prognose2030df[DATE] = prognose2030df[DATE].map(lambda x: x.replace(year=2030) if not (x.month == 2 and x.day == 29) else x.replace(month=2, day=28, year=2030))
        prognose2030df['Verbrauch [MWh]'] = prognose2030df[CONSUMPTION] * faktor
        return prognose2030df

    def wärmepumpe5():
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
        luftWasserVerbrauch = wärmepumpeVerbrauchImJahr5(heizstunden, nennleistung, luftWasserJAZ)  # in kW/h
        erdwärmeVerbrauch = wärmepumpeVerbrauchImJahr5(heizstunden, nennleistung, erdwärmeJAZ)  # in kW/h

        luftWasserVerhältnisAnzahl = verhältnisAnzahl5(wärmepumpeAnzahl2030, luftWasserVerhältnis)
        erdwärmeVerhältnisAnzahl = verhältnisAnzahl5(wärmepumpeAnzahl2030, erdwärmeVerhältnis)

        return luftWasserVerbrauch * luftWasserVerhältnisAnzahl + erdwärmeVerbrauch * erdwärmeVerhältnisAnzahl  # kWh

    # berechnung des Verbrauchs einer Wärmepumpe im Jahr
    def wärmepumpeVerbrauchImJahr5(heizstunden, nennleistung, jaz): 
        return (heizstunden * nennleistung) / jaz # (Heizstunden * Nennleistung) / JAZ = Stromverbrauch pro Jahr

    def verhältnisAnzahl5(wärmepumpeAnzahl2030, verhältnis):
        return wärmepumpeAnzahl2030 * verhältnis


    def eMobilität5():
        highECars = 15000000
        lowECars = 8000000
        middleECars = 11500000

        eMobilität2030 = lowECars  # 15mio bis 20230
        eMobilitätBisher = 1307901  # 1.3 mio
        verbrauchPro100km = 21  # 21kWh
        kilometerProJahr = 15000  # 15.000km

        eMobilitätVerbrauch = (verbrauchPro100km / 100) * kilometerProJahr  # kWh

        return (eMobilität2030 - eMobilitätBisher) * eMobilitätVerbrauch

    def faktorRechnung5(verbrauch2022df, eMobilitätHochrechnung2030):
        gesamtVerbrauch2022 = (otherFactors5(verbrauch2022df))*1000000000 + 504515946000 # mal1000 weil MWh -> kWh
        return (gesamtVerbrauch2022  + eMobilitätHochrechnung2030) / (504515946000) #ges Verbrauch 2021

    def prognoseRechnung5(verbrauch2022df, faktor):
        verbrauch2030df = verbrauch2022df['Verbrauch [kWh]'] * faktor
        return verbrauch2030df

    def otherFactors5( verbrauch2022df):
        #indHigh = (wärmepumpeHochrechnung2030*(1+3/7))*(72/26)
        indLow = verbrauch2022df[CONSUMPTION].sum()*0.45*0.879/1000000
        indMiddle = 0

        # positive Faktoren
        railway = 5  # TWh
        powerNetLoss = 1
        industry = indLow

        # negative Faktoren
        efficiency = 51
        other = 6

        return railway  + powerNetLoss - efficiency - other + industry/1000000000
#--------------------------------------------------------------------------
# GOOD SCENARIO WITH LOAD PROFILE METHODS END
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# MEDIUM SCENARIO METHODS BEGIN
#--------------------------------------------------------------------------
    def process_and_plot_2030_dataMi(production_df, consumption_df, load_profile_df, selected_date):
            
        # Realistisches Ausbau (based on frauenhofer) Szenario 2030 basierend auf 2022 Wetter (mittleres Wetter) ((2021 wäre schlechtes Wetter))
        production_2022df = production_df[production_df[DATE].dt.year == 2022]
        prognoseErzeugung2030_realistic_2022_df = production_2022df.copy()
        prognoseErzeugung2030_realistic_2022_df[DATE] = prognoseErzeugung2030_realistic_2022_df[DATE].map(lambda x: x.replace(year=2030) if not (x.month == 2 and x.day == 29) else x.replace(month=2, day=28, year=2030))

        windonshore_2030_factor_2022_realistic = 1.2921  # 
        windoffshore_2030_factor_2022_realistic = 2.13621  # 
        pv_2030_factor_2022_realistic = 1.821041  # assumig PV will increase by 182%

        def scale_2030_factors(df,windonshore_2030_factor_2022_realistic,windoffshore_2030_factor_2022_realistic,
                                            pv_2030_factor_2022_realistic):
            df_copy = df.copy()
            df_copy[WIND_ONSHORE] *= windonshore_2030_factor_2022_realistic
            df_copy[WIND_OFFSHORE] *= windoffshore_2030_factor_2022_realistic
            df_copy[PHOTOVOLTAIC] *= pv_2030_factor_2022_realistic
            df_copy['Total Production'] = df_copy[[BIOMAS, HYDROELECTRIC, WIND_OFFSHORE, WIND_ONSHORE, PHOTOVOLTAIC, OTHER_RENEWABLE]].sum(axis=1)
            return df_copy

        # Scale the data by the factors
        scaled_production_df = scale_2030_factors(prognoseErzeugung2030_realistic_2022_df, windonshore_2030_factor_2022_realistic,windoffshore_2030_factor_2022_realistic,
                                            pv_2030_factor_2022_realistic)

        # Filter the data for the selected date
        scaled_selected_production_df = scaled_production_df[scaled_production_df[DATE] == selected_date]

        verbrauch2030df = energyConsumption2(consumption_df)

        selected_consumption2030df = verbrauch2030df[verbrauch2030df[DATE] == selected_date]
        scaled_selected_production_df = scaled_selected_production_df[scaled_selected_production_df[DATE] == selected_date]
        
        plot_energy_data(selected_consumption2030df, scaled_selected_production_df, selected_date) # Plot the data
        plot_renewable_percentage(scaled_production_df, verbrauch2030df) # Plot the renewable percentage


        #------------------------------------
        # TEST with yearly metrics

        year = selected_date.year
        
        # Filter the production_df dataframe for the selected year
        filtered_production_df = scaled_production_df[scaled_production_df[DATE].dt.year == year]
        filtered_consumption_df = verbrauch2030df[verbrauch2030df[DATE].dt.year == year]

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

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label='Total Renewable Energy Production (TWh)', value=f'{total_ree_sum * M_to_TWh:.2f}')

        with col2:
            # st.metric(label='Total Consumption (TWh)', value=f'{total_consumption_y * M_to_TWh:.2f}')
            st.metric(label='Total Consumption (TWh)', value=560.8)

        #------------------------------------

        return scaled_production_df, verbrauch2030df

    # Funktion zur Berechnung und Anzeige der aggregierten Daten pro Jahr
    # Author: Bjarne, Noah
    def energyConsumption2(consumption_df):
        wärmepumpeHochrechnung2030 = wärmepumpe2()
        eMobilitätHochrechnung2030 = eMobilität2()

        verbrauch2022df = consumption_df[consumption_df[DATE].dt.year == 2022]
        prognose2030df = verbrauch2022df.copy()
        faktor = faktorRechnung2(verbrauch2022df, wärmepumpeHochrechnung2030, eMobilitätHochrechnung2030)

        prognose2030df[DATE] = prognose2030df[DATE].map(lambda x: x.replace(year=2030) if not (x.month == 2 and x.day == 29) else x.replace(month=2, day=28, year=2030))

        prognose2030df['Verbrauch [MWh]'] = prognose2030df[CONSUMPTION] * faktor

        combined_df = pd.concat([verbrauch2022df[[STARTTIME, CONSUMPTION]], prognose2030df[['Verbrauch [MWh]']]], axis=1)

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
        indLow = verbrauch2022df[CONSUMPTION].sum()*0.45*0.879/1000000
        indMiddle = 0

        # positive Faktoren
        railway = 5  # TWh
        powerNetLoss = 1
        industry = indMiddle

        # negative Faktoren
        other = 6

        return railway  + powerNetLoss  - other + industry/1000000000

#--------------------------------------------------------------------------
# MEDIUM SCENARIO METHODS END
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# PESSIMISTIC SCENARIO METHODS BEGIN
#--------------------------------------------------------------------------
    def process_and_plot_2030_dataSchlecht(production_df, consumption_df, load_profile_df, selected_date):
        
        # Realistisches Ausbau (based on frauenhofer) Szenario 2030 basierend auf 2022 Wetter (mittleres Wetter) ((2021 wäre schlechtes Wetter))
        production_2022df = production_df[production_df[DATE].dt.year == 2022]
        prognoseErzeugung2030_realistic_2022_df = production_2022df.copy()
        prognoseErzeugung2030_realistic_2022_df[DATE] = prognoseErzeugung2030_realistic_2022_df[DATE].map(lambda x: x.replace(year=2030) if not (x.month == 2 and x.day == 29) else x.replace(month=2, day=28, year=2030))

        windonshore_2030_factor_2022_realistic = 1.2921  # 
        windoffshore_2030_factor_2022_realistic = 2.13621  # 
        pv_2030_factor_2022_realistic = 1.821041  # assumig PV will increase by 182%

        def scale_2030_factors(df,windonshore_2030_factor_2022_realistic,windoffshore_2030_factor_2022_realistic,
                                            pv_2030_factor_2022_realistic):
            df_copy = df.copy()
            df_copy[WIND_ONSHORE] *= windonshore_2030_factor_2022_realistic
            df_copy[WIND_OFFSHORE] *= windoffshore_2030_factor_2022_realistic
            df_copy[PHOTOVOLTAIC] *= pv_2030_factor_2022_realistic
            df_copy['Total Production'] = df_copy[[BIOMAS, HYDROELECTRIC, WIND_OFFSHORE, WIND_ONSHORE, PHOTOVOLTAIC, OTHER_RENEWABLE]].sum(axis=1)
            return df_copy

        # Scale the data by the factors
        scaled_production_df = scale_2030_factors(prognoseErzeugung2030_realistic_2022_df, windonshore_2030_factor_2022_realistic,windoffshore_2030_factor_2022_realistic,
                                            pv_2030_factor_2022_realistic)

        # Filter the data for the selected date
        scaled_selected_production_df = scaled_production_df[scaled_production_df[DATE] == selected_date]

        verbrauch2030df = energyConsumption1(consumption_df)

        selected_consumption2030df = verbrauch2030df[verbrauch2030df[DATE] == selected_date]
        scaled_selected_production_df = scaled_selected_production_df[scaled_selected_production_df[DATE] == selected_date]

        plot_energy_data(selected_consumption2030df, scaled_selected_production_df, selected_date)
        plot_renewable_percentage(scaled_production_df, verbrauch2030df)


        #------------------------------------
        # TEST with yearly metrics

        year = selected_date.year
        
        # Filter the production_df dataframe for the selected year
        filtered_production_df = scaled_production_df[scaled_production_df[DATE].dt.year == year]
        filtered_consumption_df = verbrauch2030df[verbrauch2030df[DATE].dt.year == year]

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

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label='Total Renewable Energy Production (TWh)', value=f'{total_ree_sum * M_to_TWh:.2f}')

        with col2:
            # st.metric(label='Total Consumption (TWh)', value=f'{total_consumption_y * M_to_TWh:.2f}')
            st.metric(label='Total Consumption (TWh)', value=659.3)

        #------------------------------------

        return scaled_production_df, verbrauch2030df

    # Funktion zur Berechnung und Anzeige der aggregierten Daten pro Jahr
    # Author: Bjarne, Noah
    def energyConsumption1(consumption_df):
        wärmepumpeHochrechnung2030 = wärmepumpe1()
        eMobilitätHochrechnung2030 = eMobilität1()

        verbrauch2022df = consumption_df[consumption_df[DATE].dt.year == 2022]
        prognose2030df = verbrauch2022df.copy()
        faktor = faktorRechnung1(verbrauch2022df, wärmepumpeHochrechnung2030, eMobilitätHochrechnung2030)

        prognose2030df[DATE] = prognose2030df[DATE].map(lambda x: x.replace(year=2030) if not (x.month == 2 and x.day == 29) else x.replace(month=2, day=28, year=2030))

        prognose2030df['Verbrauch [MWh]'] = prognose2030df[CONSUMPTION] * faktor

        combined_df = pd.concat([verbrauch2022df[[STARTTIME, CONSUMPTION]], prognose2030df[['Verbrauch [MWh]']]], axis=1)

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
        indLow = verbrauch2022df[CONSUMPTION].sum()*0.45*0.879/1000000
        indMiddle = 0

        # positive Faktoren
        railway = 5  # TWh
        powerNetLoss = 1
        industry = indHigh

        # negative Faktoren
        efficiency = 51
        other = 6

        return railway  + powerNetLoss - efficiency - other + industry/1000000000
#--------------------------------------------------------------------------
# PESSIMISTIC SCENARIO METHODS END
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
# STORAGE METHODS BY TIMO BEGIN
#--------------------------------------------------------------------------

    def powerOfStorage(expected_yearly_consumption, expected_yearly_production,prozent):
        # Kopien der DataFrames erstellen, um den Originalinhalt nicht zu verändern
        verbrauch_copy = expected_yearly_consumption.copy()
        erzeugung_copy = expected_yearly_production.copy()
      
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
  
    def powerOfStorageforsurplus(expected_yearly_consumption, expected_yearly_production,prozent):
        # Kopien der DataFrames erstellen, um den Originalinhalt nicht zu verändern
        verbrauch_copy = expected_yearly_consumption.copy()
        erzeugung_copy = expected_yearly_production.copy()
      
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



    def capacity(verbrauch_df, erzeugung_df, prozent, start_capacity):
        verbrauch_copy = verbrauch_df.copy()
        erzeugung_copy = erzeugung_df.copy()
        capacity_value = 0
        energieSurPlus = 0
        efficencie = 0.9

        erzeugung_copy.set_index('Datum', inplace=True)
        verbrauch_copy.set_index('Datum', inplace=True)

        differenz = (verbrauch_copy['Verbrauch [MWh]']) * prozent - erzeugung_copy['Total Production']
        differenz_data = pd.DataFrame({'Differenz': differenz})

        total_consumption = verbrauch_copy['Verbrauch [MWh]'].sum()
        total_production = erzeugung_copy['Total Production'].sum()
        
        percentage = (total_production / total_consumption)
        
        if percentage <= prozent:
            percentage = percentage*100
            print(f"Die Erzeugung kann den angegebenen Verbrauch nicht decken ({percentage}%).")
            return 0, 0, 0
        else:
            while capacity_value == 0:
                start_capacity += 1000000
                capacity_value = start_capacity
                
                energieSurPlus = 0

                for index, value in differenz_data.iterrows():
                    if value['Differenz'] > 0:
                        if capacity_value - value['Differenz'] < 0:
                            remaining_capacity = capacity_value
                            capacity_value = 0
                            print(f"Speicher ist leer, es konnten nur {capacity_value} MWh entnommen werden.")
                            break
                        else:
                            capacity_value -= value['Differenz']

                    elif value['Differenz'] < 0:
                        if capacity_value + (abs(value['Differenz']) * efficencie) > start_capacity:
                            energieSurPlus = capacity_value + abs(value['Differenz']) * efficencie - start_capacity
                            capacity_value = start_capacity
                        else:
                            capacity_value -= (value['Differenz'] * efficencie)
            
            return capacity_value, start_capacity, energieSurPlus




    def investmentcost(capacity_needed):   #Eventuell noch prozente von Speicherarten hinzufügen
        capacity_in_germany=0  
        cost_of_Battery=100 #Einheit sind Euro/kWh

        capacity_for_expension=capacity_needed-capacity_in_germany

        price=(cost_of_Battery*capacity_for_expension)/(1000000) #Price in Bilion

        print(f"Der Preis in Milliarden beträgt:{price}")


    #----------------------------------
    # Power Calculation

    def calculate_and_plot_power_storage_surplus(expected_yearly_production, expected_yearly_consumption):

        # Verwendung der Funktion mit den entsprechenden DataFrames verbrauch2030df und scaled_production_df
        result_differenz_sorted_80, power_in_GW_80 = powerOfStorage(expected_yearly_consumption, expected_yearly_production,0.8)
        result_differenz_sorted_90, power_in_GW_90 = powerOfStorage(expected_yearly_consumption, expected_yearly_production,0.9)
        result_differenz_sorted_100, power_in_GW_100 = powerOfStorage(expected_yearly_consumption, expected_yearly_production,1)

        #Benötigte Leistung für den Überschuss
        result_differenz_sorted_surplus_80, power_in_GW_surplus_80 = powerOfStorageforsurplus(expected_yearly_consumption, expected_yearly_production,0.8)
        result_differenz_sorted_surplus_90, power_in_GW_surplus_90 = powerOfStorageforsurplus(expected_yearly_consumption, expected_yearly_production,0.9)
        result_differenz_sorted_surplus_100, power_in_GW_surplus_100 = powerOfStorageforsurplus(expected_yearly_consumption, expected_yearly_production,1)

            # Create a DataFrame with the power values for consumption and surplus
        df = pd.DataFrame({
            'Percentage': ['80%', '90%', '100%'],
            'Power in GW (Consumption)': [power_in_GW_80, power_in_GW_90, power_in_GW_100],
            'Power in GW (Surplus)': [power_in_GW_surplus_80, power_in_GW_surplus_90, power_in_GW_surplus_100]
        })

        percentages = ['80%', '90%', '100%']

        fig = go.Figure(data=[
            go.Bar(name='Power in GW (Consumption)', x=percentages, y=[power_in_GW_80, power_in_GW_90, power_in_GW_100]),
            go.Bar(name='Power in GW (Surplus)', x=percentages, y=[power_in_GW_surplus_80, power_in_GW_surplus_90, power_in_GW_surplus_100])
        ])

        fig.update_layout(
            title='Required power of storage [GW] for 80-100 % coverage of consumption and surplus',
            xaxis=dict(title='Covered Consumption by storage [%]', tickmode='array', tickvals=[80, 90, 100]),
            yaxis=dict(title='Required power of storage [GW]'))


        fig.update_layout(barmode='group')
        fig.update_traces(width=3)  # Adjust the width as per your preference (0.5 is an example)
        st.plotly_chart(fig)

        st.dataframe(df)



    #----------------------------------
    # Capacity Calculation

    def calculate_and_plot_storage_capacity(expected_yearly_production, expected_yearly_consumption, scenario_to_plot):

        st.subheader('Storage Capacity')

        # acutal calculation

        # capacity_value_80, capacity_value_start_80,energieSurPlus_80  = capacity(expected_yearly_consumption, expected_yearly_production, 0.8, 10000000)
        # capacity_value_90, capacity_value_start_90,energieSurPlus_90 = capacity(expected_yearly_consumption, expected_yearly_production, 0.9, 10000000)
        # capacity_value_100, capacity_value_start_100,energieSurPlus_100 = capacity(expected_yearly_consumption, expected_yearly_production, 1, 10000000)

        # pre entered values for performance
        if scenario_to_plot == 'good':
            capacity_value_80, capacity_value_start_80,energieSurPlus_80  = 0,11000000,0
            capacity_value_90, capacity_value_start_90,energieSurPlus_90 = 0,11000000,0
            capacity_value_100, capacity_value_start_100,energieSurPlus_100 = 0,13000000,0
        elif scenario_to_plot == 'mid':
            capacity_value_80, capacity_value_start_80,energieSurPlus_80  = 0,0,0
            capacity_value_90, capacity_value_start_90,energieSurPlus_90 = 0,0,0
            capacity_value_100, capacity_value_start_100,energieSurPlus_100 = 0,0,0
        elif scenario_to_plot == 'bad':
            capacity_value_80, capacity_value_start_80,energieSurPlus_80  = 0,0,0
            capacity_value_90, capacity_value_start_90,energieSurPlus_90 = 0,0,0
            capacity_value_100, capacity_value_start_100,energieSurPlus_100 = 0,0,0

        print(capacity_value_start_80)
        print(capacity_value_start_90)
        print(capacity_value_start_100)
        # investment_cost_80 = investmentcost(capacity_value_start_80)

        df = pd.DataFrame({
            'Percentage': ['80%', '90%', '100%'],
            'Capacity Start Value in TWh': [capacity_value_start_80 * M_to_TWh, capacity_value_start_90 * M_to_TWh, capacity_value_start_100 * M_to_TWh]
        })

        y_values = [capacity_value_start_80 * M_to_TWh, capacity_value_start_90 * M_to_TWh, capacity_value_start_100 * M_to_TWh]

        fig = go.Figure(data=[
            go.Bar(name='Capacity in MWh (Consumption)', x=['80%', '90%', '100%'], y=y_values 
            )
        ])

        fig.update_layout(
            title='Required capacity of storage [TWh] for 80-100 % coverage of consumption and surplus',
            xaxis=dict(title='Covered consumption by storage [%]', tickmode='array', tickvals=[80, 90, 100]),
            yaxis=dict(title='Required capacity [TWh]')
        )

        fig.update_traces(width=3)  # Adjust the width as per your preference (0.5 is an example)
        st.plotly_chart(fig)
        st.dataframe(df)


#--------------------------------------------------------------------------
# STORAGE METHODS BY TIMO END
#--------------------------------------------------------------------------


    # assumptions = [
    #     "Consumption of EVs",
    #     "Consumption of heat pumps",
    #     "Consumption of railway",
    #     "Consumption of battery production and server rooms",
    #     "Energy loss in the power grid",
    #     "Efficiency of the power plants"
    # ]

    # delta_values = [
    #     43.13, 
    #     32.82,
    #     5,
    #     13,
    #     -1,
    #     -51
    # ]

    # colors = ['green' if delta > 0 else 'red' for delta in delta_values]

    # fig_4 = go.Figure(data=go.Bar(x=assumptions, y=delta_values, marker=dict(color=colors)))
    # fig_4.update_layout(
    #     title='Expected increase in consumption till 2030 compared to 2021',
    #     xaxis=dict(title='Factors'),
    #     yaxis=dict(title='Energy consumption in TWh'),
    # )

    # # Display the chart in Streamlit
    # st.plotly_chart(fig_4)

    # st.markdown("#### Choose your consumption scenario")

    # consumption_selection_slider = st.slider('Choose the consumption for 2030 in TWh', 500, 750, 800)
    # st.write("You chose", consumption_selection_slider, "TWh of consumption in 2030")

#--------------------------------------------------------------------------
# SCENARIOs STREAMLIT DISPLAY BEGIN
#--------------------------------------------------------------------------


    # Display the metrics and delta values
    st.header('Preset Scenarios')
    st.write('On this page we gathered a preset optimistic, moderate and pessimistic scenario for the production, consumption and storage solutions in 2030.')

    col1_in, col2_in, col3_in = st.columns(3)

    with col1_in:

        st.subheader('Optimistic Scenario')

        st.markdown('##### Optimistic Production Assumptions')

        st.markdown("""
                    * High generation
                    * All expansion targets of the BMWK are being achieved
                    * Good weather year for renewable energies (2020 scaled up)
                    """)        

        st.markdown('##### Optimistic Consumption Assumptions')

        st.markdown("""
                    * Low consumption
                    * E-car and heat pump targets are not being met
                    * Industrial consumption is declining
                    """)

    with col2_in:

        st.subheader('Moderate Scenario')

        st.markdown('##### Moderate Production Assumptions')

        st.markdown("""
                    * Reduced generation
                    * Expansion estimate by the Frauenhofer Institute
                    + Moderately good weather year for renewable energies (2022 scaled up)
                    """)

        st.markdown('##### Moderate Consumption Assumptions')

        st.markdown("""
                    * Average consumption​
                    * Electric vehicle and heat pump targets are not fully achieved​
                    * Industrial consumption remains the same
                    """)

    with col3_in:

        st.subheader('Pessimistic Scenario')

        st.markdown('##### Pessimistic Production Assumptions')

        st.markdown("""
                    * Lower production
                    * Expansion estimate by the Frauenhofer Institute1
                    * Medium-good weather year for renewable energies (scaled up to 2022)
                    * Same production scenario chosen (otherwise, production would be too low)
                    """)

        st.markdown('##### Pessimistic Consumption Assumptions')

        st.markdown("""
                    * High consumption
                    * E-mobility and heat pump targets are being achieved
                    * The share of industrial consumption compared to total consumption in Germany remains the same
                    """)



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


    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader('Optimistic Scenario')
        
        load_profile_df = read_load_profile('Lastprofile_SWKiel.xls')
        date = selected_date

        # GOOD SCENARIO Function Call
        expected_yearly_production, expected_yearly_consumption = process_and_plot_2030_dataGut(production_df, consumption_df, load_profile_df, date)
        st.subheader('Optimistic Storage Scenario')
        calculate_and_plot_power_storage_surplus(expected_yearly_production, expected_yearly_consumption)
        calculate_and_plot_storage_capacity(expected_yearly_production, expected_yearly_consumption, 'good')

    with col2:
        st.subheader('Medium Scenario')
        # MEDIUM SCENARIO Function Call
        expected_yearly_production, expected_yearly_consumption = process_and_plot_2030_dataMi(production_df, consumption_df, load_profile_df, date)
        st.subheader('Medium Storage Scenario')
        calculate_and_plot_power_storage_surplus(expected_yearly_production, expected_yearly_consumption)
        calculate_and_plot_storage_capacity(expected_yearly_production, expected_yearly_consumption, 'mid')

    with col3:
        st.subheader('Pessimistic Scenario')
        # PESSIMISTIC SCENARIO Function Call
        expected_yearly_production, expected_yearly_consumption = process_and_plot_2030_dataSchlecht(production_df, consumption_df, load_profile_df, date)
        st.subheader('Pessimistic Storage Scenario')
        calculate_and_plot_power_storage_surplus(expected_yearly_production, expected_yearly_consumption)
        calculate_and_plot_storage_capacity(expected_yearly_production, expected_yearly_consumption, 'bad')

#--------------------------------------------------------------------------
# SCENARIOs STREAMLIT DISPLAY END
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# LOAD PROFILE SCENARIOs STREAMLIT DISPLAY BEGIN
#--------------------------------------------------------------------------
with tab3:

    st.header('Load Profile Scenario')
    st.write('In this scenario we used our preset optimistic scenario and added a heat pump load profile from the SWKiel GmbH, to make the scenario more realistic')

    start_date_load = datetime.date(2030, 1, 1)
    end_date_load = datetime.date(2030, 12, 31)
    default_date_load = datetime.date(2030, 1, 1)
    st.write("##")
    input_date_load = st.date_input(
        "Select a Date",
        value=default_date_load, 
        min_value=start_date_load,
        max_value=end_date_load,
        format="DD.MM.YYYY",
        key='date_input_load'  # Add a unique key argument
    )
    selected_date_load = pd.to_datetime(
        input_date_load,
        format="%d.%m.%Y",
    )

    date_load = selected_date_load

    # TEST with Good Scenario with load profile
    process_and_plot_2030_dataGut2(production_df, consumption_df, load_profile_df, date_load)

#--------------------------------------------------------------------------
# LOAD PROFILE SCENARIOs STREAMLIT DISPLAY END
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# BUILD your Scenario BEGIN
#--------------------------------------------------------------------------

with tab4:

    def process_and_plot_2030_dataGut_IND(production_df, consumption_df, load_profile_df, selected_date, heatpumpamount,evamount):
        
        # POSITIVE SCENARIO Production based on 2020 and BMWK goals
        production_2020df = production_df[production_df[DATE].dt.year == 2020]
        prognoseErzeugung2030_positive_df = production_2020df.copy()
        #prognoseErzeugung2030_positive_df['Date'] = prognoseErzeugung2030_positive_df['Date'].map(lambda x: x.replace(year=2030))
        prognoseErzeugung2030_positive_df[DATE] = prognoseErzeugung2030_positive_df[DATE].map(
        lambda x: x.replace(year=2030) if not (x.month == 2 and x.day == 29) else x.replace(month=2, day=28, year=2030))

        windonshore_2030_factor_2020_positive = 2.13589  # 
        windoffshore_2030_factor_2020_postive = 3.92721  #
        pv_2030_factor_2020_postive = 4.2361193  # assumig PV will increase by 423%

        # Scale the data by the factors
        scaled_production_df = scale_2030_factors(prognoseErzeugung2030_positive_df, windonshore_2030_factor_2020_positive,windoffshore_2030_factor_2020_postive,
                                            pv_2030_factor_2020_postive)

        # Filter the data for the selected date
        scaled_selected_production_df = scaled_production_df[scaled_production_df[DATE] == selected_date]

        verbrauch2030df = energyConsumption_IND(consumption_df, heatpumpamount,evamount)

        print("+++++++++++++++++++++++++++")
        print("Verbrauch 2030:", verbrauch2030df['Verbrauch [MWh]'].sum()/1000 , "TWhhusp\n")
        print("+++++++++++++++++++++++++++")

                #------------------------------------
        # TEST with yearly metrics

        year = selected_date.year
        
        # Filter the production_df dataframe for the selected year
        filtered_production_df = scaled_production_df[scaled_production_df[DATE].dt.year == 2030]
        filtered_consumption_df = verbrauch2030df[verbrauch2030df[DATE].dt.year == 2030]

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

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label='Total Renewable Energy Production (TWh)', value=f'{total_ree_sum * M_to_TWh:.2f}')
        

        with col2:
            st.metric(label='Total Consumption (TWh)', value=f"{verbrauch2030df['Verbrauch [MWh]'].sum() * M_to_TWh:.2f}")

        #------------------------------------

        selected_consumption2030df = verbrauch2030df[verbrauch2030df[DATE] == selected_date]
        scaled_selected_production_df = scaled_selected_production_df[scaled_selected_production_df[DATE] == selected_date]

        plot_energy_data(selected_consumption2030df, scaled_selected_production_df, selected_date)
        plot_renewable_percentage(scaled_production_df, verbrauch2030df)


        return scaled_production_df, verbrauch2030df

    # Funktion zur Berechnung und Anzeige der aggregierten Daten pro Jahr
    # Author: Bjarne, Noah
    def energyConsumption_IND(consumption_df, heatpumpamount, evamount):
        wärmepumpeHochrechnung2030 = wärmepumpe_IND(heatpumpamount)
        eMobilitätHochrechnung2030 = eMobilität_IND(evamount)

        verbrauch2022df = consumption_df[consumption_df[DATE].dt.year == 2020]
        prognose2030df = verbrauch2022df.copy()
        faktor = faktorRechnung(verbrauch2022df, wärmepumpeHochrechnung2030, eMobilitätHochrechnung2030)
        print("Verbr df:", prognose2030df)
        print("Faktor: ", faktor)
        # Change the year in 'Datum' column to 2030
        prognose2030df[DATE] = prognose2030df[DATE].map(lambda x: x.replace(year=2030) if not (x.month == 2 and x.day == 29) else x.replace(month=2, day=28, year=2030))

        prognose2030df['Verbrauch [MWh]'] = prognose2030df[CONSUMPTION] * faktor

        combined_df = pd.concat([verbrauch2022df[[STARTTIME, CONSUMPTION]], prognose2030df[['Verbrauch [MWh]']]], axis=1)
        print("Verbrauch 2030:", prognose2030df['Verbrauch [MWh]'].sum()/1000 , "TWhhusp\n")
        print("Consumption 2022:", prognose2030df[CONSUMPTION].sum()/1000 , "TWh\n")

        return prognose2030df

    def wärmepumpe_IND(heatpumpamount):
        highScenario = 500000
        lowScenario = 236000
        middleScenario = 368000
        wärmepumpeAnzahl2030 = heatpumpamount

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
    def wärmepumpeVerbrauchImJahr_IND(heizstunden, nennleistung, jaz): 
        return (heizstunden * nennleistung) / jaz # (Heizstunden * Nennleistung) / JAZ = Stromverbrauch pro Jahr

    def verhältnisAnzahl_IND(wärmepumpeAnzahl2030, verhältnis):
        return wärmepumpeAnzahl2030 * verhältnis


    def eMobilität_IND(evamount):
        highECars = 15000000
        lowECars = 8000000
        middleECars = 11500000

        eMobilität2030 = evamount  # 15mio bis 20230
        eMobilitätBisher = 1307901  # 1.3 mio
        verbrauchPro100km = 21  # 21kWh
        kilometerProJahr = 15000  # 15.000km

        eMobilitätVerbrauch = (verbrauchPro100km / 100) * kilometerProJahr  # kWh

        return (eMobilität2030 - eMobilitätBisher) * eMobilitätVerbrauch

    def faktorRechnung_IND(verbrauch2022df, wärmepumpeHochrechnung2030, eMobilitätHochrechnung2030):
        gesamtVerbrauch2022 = (otherFactors(wärmepumpeHochrechnung2030, verbrauch2022df))*1000000000 + 504515946000 # mal1000 weil MWh -> kWh
        return (gesamtVerbrauch2022 + wärmepumpeHochrechnung2030 + eMobilitätHochrechnung2030) / (504515946000) #ges Verbrauch 2021

    def prognoseRechnung_IND(verbrauch2022df, faktor):
        verbrauch2030df = verbrauch2022df['Verbrauch [kWh]'] * faktor
        return verbrauch2030df

    def otherFactors_IND(wärmepumpeHochrechnung2030, verbrauch2022df):
        indHigh = (wärmepumpeHochrechnung2030*(1+3/7))*(72/26)
        indLow = -verbrauch2022df[CONSUMPTION].sum()*1000*0.45*0.121
        indMiddle = 0

        # positive Faktoren
        railway = 5  # TWh
        powerNetLoss = 1
        industry = indLow

        # negative Faktoren
        efficiency = 51
        other = 6

        return railway  + powerNetLoss - other + industry/1000000000


    st.header("Build your own scenario")
    st.write("This is our first approach to a scenario builder. With this tool you can build your own scenario, by changing the amount of heatpumps and EVs in 2030 and see how the plots change. We are currently working on adding more customizable parameters.")

    st.markdown('#### Choose the amount of heatpumps in 2030')
    heatpump_selection_slider = st.slider('Choose the amount of heatpumps in 2030', 1500000, 4000000, 2500000)
    st.write("You chose", heatpump_selection_slider, "for the amount of heat pumps in 2030")

    st.markdown('#### Choose the amount of EVs in 2030')
    ev_selection_slider = st.slider('Choose the amount of EVs in 2030', 5000000, 20000000, 11000000)
    st.write("You chose", ev_selection_slider, "for the amount of heat pumps in 2030")

    start_date_load = datetime.date(2030, 1, 1)
    end_date_load = datetime.date(2030, 12, 31)
    default_date_load = datetime.date(2030, 1, 1)
    st.write("##")
    input_date_load = st.date_input(
        "Select a Date",
        value=default_date_load, 
        min_value=start_date_load,
        max_value=end_date_load,
        format="DD.MM.YYYY",
        key='date_input_ind'  # Add a unique key argument
    )
    selected_date_load = pd.to_datetime(
        input_date_load,
        format="%d.%m.%Y",
    )

    date_ind = date

    if st.button('Process and Plot'):
        process_and_plot_2030_dataGut_IND(production_df, consumption_df, load_profile_df, selected_date_load, heatpump_selection_slider, ev_selection_slider)



#--------------------------------------------------------------------------
# BUILD your Scenario END
#--------------------------------------------------------------------------



#---------------------------------------------------------
# STORAGE PRINT OUTS TIMO BEGIN
#---------------------------------------------------------


# # Ausgabe des sortierten DataFrames
# print("Sortiertes DataFrame nach Differenz:")
# print(result_differenz_sorted_80)

# # Ausgabe des Mittelwerts der ersten 100 größten Differenzen
# print("Leistung der Speicher für 80% Dekung in GW:", power_in_GW_80)

#     # Verwendung der Funktion mit den entsprechenden DataFrames verbrauch2030df und scaled_production_df

# # Ausgabe des sortierten DataFrames
# print("Sortiertes DataFrame nach Differenz:")
# print(result_differenz_sorted_90)

# # Ausgabe des Mittelwerts der ersten 100 größten Differenzen
# print("Leistung der Speicher für 90% Dekung in GW:", power_in_GW_90)

# # Verwendung der Funktion mit den entsprechenden DataFrames verbrauch2030df und scaled_production_df
# result_differenz_sorted_100, power_in_GW_100 = powerOfStorage(verbrauch2030df, scaled_production_df,1)

# # Ausgabe des sortierten DataFrames
# print("Sortiertes DataFrame nach Differenz:")
# print(result_differenz_sorted_100)

# # Ausgabe des Mittelwerts der ersten 100 größten Differenzen
# print("Leistung der Speicher für 100% Dekung in GW:", power_in_GW_100) 

# #Benötigte Leistung für den Überschuss

# # Verwendung der Funktion mit den entsprechenden DataFrames verbrauch2030df und scaled_production_df
# result_differenz_sorted_surplus, power_in_GW_surplus = powerOfStorageforsurplus(verbrauch2030df, scaled_production_df,0.8)

# # Ausgabe des sortierten DataFrames
# print("Sortiertes DataFrame nach Differenz:")
# print(result_differenz_sorted_surplus)

# # Ausgabe des Mittelwerts der ersten 100 größten Differenzen
# print("Leistung der Speicher für die Überschussaufnahme in GW:", power_in_GW_surplus)

#---------------------------------------------------------
# STORAGE PRINT OUTS TIMO END
#---------------------------------------------------------