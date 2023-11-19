import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import datetime
import time
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# Apply dark background style
style.use('dark_background')

st.title("WATT-Meister-Consulting Calculator")
st.divider()
st.subheader('Energy production and consumption')
st.write('For the following plots, we collected the electricity market data of Germany for the years 2020, 2021, and 2022 and analyzed the production and consumption. In the first plot, you can see the production and consumption for any specific day in the period from 2020 to 2022.')

start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2022, 12, 31)
default_date = datetime.date(2020, 1, 1)
st.write("##")
input_date = st.date_input("Select a Date",value = default_date, min_value=start_date, max_value=end_date)
selected_date = input_date.strftime("%d.%m.%Y") 


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



# Create the layout for the chart with custom x-axis ticks
layout = go.Layout(
    title=f'Renewable Energy Production and Total Consumption on {selected_date}',
    xaxis=dict(
        title='Time (hours)',
        tickvals=pd.date_range(selected_production[STARTTIME].min(), selected_production[STARTTIME].max(), freq='H'),
        ticktext=[i.strftime('%H') for i in pd.date_range(selected_production[STARTTIME].min(), selected_production[STARTTIME].max(), freq='H')],
        type='category',
        showgrid=True
    ),
    yaxis=dict(
        title='Energy (MWh)',
        showgrid=True  # Display grid for the y-axis
    ),
    showlegend=True,
    legend=dict(
        x=0,
        y=1,
        traceorder='normal',
        bgcolor='rgba(0,0,0,0)'
    )
)

# Create the figure
fig = go.Figure(layout=layout)

# Add the traces
fig.add_trace(go.Scatter(
    x=selected_consumption[STARTTIME],
    y=selected_consumption[CONSUMPTION],
    mode='none',
    fill='tozeroy',  # Fill area to the x-axis
    fillcolor='rgba(254, 255, 178, .9)',  # Fill color for consumption trace
    name='Total Consumption',
    # marker=dict(color='#FF0000'),  # Red line color
    showlegend=True,

))

fig.add_trace(go.Scatter(
    x=selected_production[STARTTIME],
    y=selected_production['Total Production'],
    mode='none',
    fill='tozeroy',
    fillcolor='rgba(142, 211, 199, .9)',
    name='Total Production',
    # marker=dict(color='#FF0000'),  # Red line color
    showlegend=True
))



# Display the figure in Streamlit
st.plotly_chart(fig)







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

