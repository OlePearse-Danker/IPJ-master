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


# Apply dark background style
style.use('dark_background')

st.title("WATT-Meister-Consulting Calculator")
st.divider()
st.subheader('Energy production and consumption')
st.write('For the following plots, we collected the electricity market data of Germany for the years 2020, 2021, and 2022 and analyzed the production and consumption. In the first plot, you can see the production and consumption for any specific day in the period from 2020 to 2022.')

start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2022, 12, 31)
default_date = datetime.date(2020, 1, 3)
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
        y=selected_consumption[CONSUMPTION],
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
        y=selected_production['Total Production'],
        mode='none',
        name='Total Renewable Production',
        fill='tozeroy',
        fillcolor='#0D7F9F'
    )
)


fig_1.update_layout(
    title=f'Energy Production and Consumption on {selected_date}',
    xaxis=dict(title='Time (hours)', showgrid=True),
    yaxis=dict(title='Energy (MWh)', showgrid=True),
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

# Compute the total production for each renewable energy type
total_biomass = filtered_production_df[BIOMAS].sum()
total_waterpower = filtered_production_df[HYDROELECTRIC].sum()
total_windoff = filtered_production_df[WIND_OFFSHORE].sum()
total_windon = filtered_production_df[WIND_ONSHORE].sum()
total_pv = filtered_production_df[PHOTOVOLTAIC].sum()
total_other_ree = filtered_production_df[OTHER_RENEWABLE].sum()

# Create a bar chart to display the yearly production of renewable energy
fig_3 = px.bar(x=[BIOMAS, HYDROELECTRIC, WIND_OFFSHORE, WIND_ONSHORE, PHOTOVOLTAIC, OTHER_RENEWABLE],
             y=[total_biomass, total_waterpower, total_windoff, total_windon, total_pv, total_other_ree],
             labels={'x': 'Renewable Energy Type', 'y': 'Total Production (MWh)'},
             title=f"Yearly Production of Renewable Energy - {year}")

fig_3.data[0].x = ['biomass', 'hydroelectric', 'wind offshore', 'wind onshore', 'photovoltaic', 'other_renewable']

# Display the bar chart using st.plotly_chart
st.plotly_chart(fig_3)

# ------------------------------------------------

# first forecast by Lucas

st.subheader('Forecast of Renewable Energy production')

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

# # Ein Beispiel für die Berechnung der Gesamtsumme einer bestimmten Art
# selected_energy_type = WIND_ONSHORE

# # Für einen Tag
# selected_production_day = selected_production[selected_energy_type].sum()
# print(f"{selected_energy_type} Production on {selected_date}: {selected_production_day} MWh")

# # Für ein Jahr
# selected_production_year = production_by_type_and_year.loc[selected_date.year, selected_energy_type]
# print(f"{selected_energy_type} Production for {selected_date.year}: {selected_production_year} MWh")

# # Ein Beispiel für die Arbeit mit Listen einer bestimmten Art
# selected_energy_type = WIND_ONSHORE

# # Für einen Tag
# selected_production_day_list = selected_production[selected_energy_type].astype(float).tolist()
# print(f"{selected_energy_type} Production List on {selected_date}: {selected_production_day_list}")

# # Für ein Jahr
# selected_production_year_list = production_by_type_and_year.loc[selected_date.year, selected_energy_type].tolist()
# print(f"{selected_energy_type} Production List for {selected_date.year}: {selected_production_year_list}")