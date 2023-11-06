import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib.style as style
import datetime
import time
import streamlit as st
from matplotlib.animation import FuncAnimation
import numpy as np
import plotly.express as px



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

def parse_datetime(date_str, time_str):
    return datetime.datetime.strptime(f"{date_str} {time_str}", "%d.%m.%Y %H:%M")

startzeit = time.time()

csv_datei1 = 'Realisierte_Erzeugung_202001010000_202212312359_Viertelstunde.csv'
csv_datei2 = 'Realisierter_Stromverbrauch_202001010000_202212312359_Viertelstunde.csv'
csv_datei3 = 'Installierte Erzeugungsleistung 2020-2022.csv'


energie_daten = []
energie_daten2 = []
production = []
consumption = []

with open(csv_datei1, 'r') as file:
    csv_reader = csv.reader(file, delimiter=';')
    next(csv_reader)
    for row in csv_reader:
        datum = row[0]
        anfang = row[1]
        ende = row[2]
        biomasse = float(row[3].replace('.', '').replace(',', '.'))
        wasserkraft = float(row[4].replace('.', '').replace(',', '.'))
        wind_offshore = float(row[5].replace('.', '').replace(',', '.'))
        wind_onshore = float(row[6].replace('.', '').replace(',', '.'))
        photovoltaik = float(row[7].replace('.', '').replace(',', '.'))
        try:
            sonstige_erneuerbare = float(row[8].replace('.', '').replace(',', '.')) 
        except ValueError:
            sonstige_erneuerbare = 0.0
        kernenergie = float(row[9].replace('.', '').replace(',', '.'))
        braunkohle = float(row[10].replace('.', '').replace(',', '.'))
        steinkohle = float(row[11].replace('.', '').replace(',', '.'))
        erdgas = float(row[12].replace('.', '').replace(',', '.'))
        pumpspeicher = float(row[13].replace('.', '').replace(',', '.'))
        sonstige_konventionelle = float(row[14].replace('.', '').replace(',', '.'))

        datensatz = {
            'Datum': datum,
            'Anfang': anfang,
            'Ende': ende,
            'Biomasse [MWh]': biomasse,
            'Wasserkraft [MWh]': wasserkraft,
            'Wind Offshore [MWh]': wind_offshore,
            'Wind Onshore [MWh]': wind_onshore,
            'Photovoltaik [MWh]': photovoltaik,
            'Sonstige Erneuerbare [MWh]': sonstige_erneuerbare,
            'Kernenergie [MWh]': kernenergie,
            'Braunkohle [MWh]': braunkohle,
            'Steinkohle [MWh]': steinkohle,
            'Erdgas [MWh]': erdgas,
            'Pumpspeicher [MWh]': pumpspeicher,
            'Sonstige Konventionelle [MWh]': sonstige_konventionelle
        }
        energie_daten.append(datensatz)


with open(csv_datei2, 'r') as file:
    csv_reader = csv.reader(file, delimiter=';')
    next(csv_reader)
    for row in csv_reader:
        datum = row[0]
        anfang = row[1]
        gesamt = float(row[3].replace('.', '').replace(',', '.'))

        datensatz1 = {
            'Datum': datum,
            'Anfang': anfang,
            'Gesamt (Netzlast) [MWh]': gesamt,
        }
        energie_daten2.extend([datensatz1])



production = [datensatz['Biomasse [MWh]'] + datensatz['Wasserkraft [MWh]'] + datensatz['Wind Offshore [MWh]'] + datensatz['Wind Onshore [MWh]'] + datensatz['Photovoltaik [MWh]'] + datensatz['Sonstige Erneuerbare [MWh]'] for datensatz in energie_daten]

consumption = [datensatz1['Gesamt (Netzlast) [MWh]'] for datensatz1 in energie_daten2]
    

selected_date = input_date
filtered_data = [datensatz for datensatz in energie_daten if parse_datetime(datensatz['Datum'], datensatz['Anfang']).date() == selected_date]
filtered_data2 = [datensatz1 for datensatz1 in energie_daten2 if parse_datetime(datensatz1['Datum'], datensatz1['Anfang']).date() == selected_date]


hours = [parse_datetime(datensatz['Datum'], datensatz['Anfang']).hour + parse_datetime(datensatz['Datum'], datensatz['Anfang']).minute / 60 for datensatz in filtered_data]
production_day = [datensatz['Biomasse [MWh]'] + datensatz['Wasserkraft [MWh]'] + datensatz['Wind Offshore [MWh]'] + datensatz['Wind Onshore [MWh]'] + datensatz['Photovoltaik [MWh]'] + datensatz['Sonstige Erneuerbare [MWh]'] for datensatz in filtered_data]
consumption_day = [datensatz1['Gesamt (Netzlast) [MWh]'] for datensatz1 in filtered_data2]


def range1(array1, array2):
    if len(array1) != len(array2):
        raise ValueError("Arrays must be the same length")
    
    counts = [0] * 100

    for val1, val2 in zip(array1, array2):
        ratio = val1 / val2
        percent = int(ratio * 100)

        if percent == 100:
            counts[percent - 1] += 1
        elif 0 <= percent < 100:
            counts[percent] += 1

    return counts

counts =[]
counts = range1(production, consumption)
print("Anteile in %:")
print(counts)
print()



yearly_data = {}


for datensatz in energie_daten:
    datum = datensatz['Datum']
    year = datetime.datetime.strptime(datum, '%d.%m.%Y').year
    if year not in yearly_data:
        yearly_data[year] = {
            'Production': 0,
            'Consumption': 0,
            'Biomasse [MWh]': 0,
            'Wasserkraft [MWh]': 0,
            'Wind Offshore [MWh]': 0,
            'Wind Onshore [MWh]': 0,
            'Photovoltaik [MWh]': 0,
            'Sonstige Erneuerbare [MWh]': 0,
        }

    
    yearly_data[year]['Production'] += datensatz['Biomasse [MWh]'] + datensatz['Wasserkraft [MWh]'] + datensatz['Wind Offshore [MWh]'] + datensatz['Wind Onshore [MWh]'] + datensatz['Photovoltaik [MWh]'] + datensatz['Sonstige Erneuerbare [MWh]']
    yearly_data[year]['Biomasse [MWh]'] += datensatz['Biomasse [MWh]']
    yearly_data[year]['Wasserkraft [MWh]'] += datensatz['Wasserkraft [MWh]']
    yearly_data[year]['Wind Offshore [MWh]'] += datensatz['Wind Offshore [MWh]']
    yearly_data[year]['Wind Onshore [MWh]'] += datensatz['Wind Onshore [MWh]']
    yearly_data[year]['Photovoltaik [MWh]'] += datensatz['Photovoltaik [MWh]']
    yearly_data[year]['Sonstige Erneuerbare [MWh]'] += datensatz['Sonstige Erneuerbare [MWh]']


for datensatz2 in energie_daten2:
    datum = datensatz2['Datum']
    year = datetime.datetime.strptime(datum, '%d.%m.%Y').year
    if year in yearly_data:
        yearly_data[year]['Consumption'] += datensatz2['Gesamt (Netzlast) [MWh]']


for year, data in yearly_data.items():
    print(f"Year: {year}")
    print(f"Total Renwable Energy Production: {data['Production']} MWh")
    print(f"Total Consumption: {data['Consumption']} MWh")
    print(f"Biomasse: {data['Biomasse [MWh]']} MWh")
    print(f"Wasserkraft: {data['Wasserkraft [MWh]']} MWh")
    print(f"Wind Offshore: {data['Wind Offshore [MWh]']} MWh")
    print(f"Wind Onshore: {data['Wind Onshore [MWh]']} MWh")
    print(f"Photovoltaik: {data['Photovoltaik [MWh]']} MWh")
    print(f"Sonstige Erneuerbare: {data['Sonstige Erneuerbare [MWh]']} MWh")
    print()


def find_dates_with_high_ee_ratio(energie_daten, energie_daten2, threshold=0.9):
    dates_with_high_ratio = set()
    
    for datensatz, datensatz1 in zip(energie_daten, energie_daten2):
        total_consumption = datensatz1['Gesamt (Netzlast) [MWh]']
        total_production = (
            datensatz['Biomasse [MWh]'] +
            datensatz['Wasserkraft [MWh]'] +
            datensatz['Wind Offshore [MWh]'] +
            datensatz['Wind Onshore [MWh]'] +
            datensatz['Photovoltaik [MWh]'] +
            datensatz['Sonstige Erneuerbare [MWh]']
        )
        ratio = total_production / total_consumption
        
        if ratio > threshold:
            date = parse_datetime(datensatz['Datum'], datensatz['Anfang']).date()
            dates_with_high_ratio.add(date)
    
    sorted_dates = sorted(dates_with_high_ratio)  # Сортировка дат от старых к новым
    
    return sorted_dates





dates_with_high_ee = find_dates_with_high_ee_ratio(energie_daten, energie_daten2)

if dates_with_high_ee:
    print("Daten mit 90% EE-Anteil:")
    for date in dates_with_high_ee:
        print(date)
else:
    print("Keine Daten mit 90% EE-Anteil.")

def find_dates_with_ee_ratio(energie_daten, energie_daten2, lower_threshold=0.0, upper_threshold=1.0):
    dates_with_ee_ratio = set()
    
    for datensatz, datensatz1 in zip(energie_daten, energie_daten2):
        total_consumption = datensatz1['Gesamt (Netzlast) [MWh]']
        total_production = (
            datensatz['Biomasse [MWh]'] +
            datensatz['Wasserkraft [MWh]'] +
            datensatz['Wind Offshore [MWh]'] +
            datensatz['Wind Onshore [MWh]'] +
            datensatz['Photovoltaik [MWh]'] +
            datensatz['Sonstige Erneuerbare [MWh]']
        )
        ratio = total_production / total_consumption
        
        if lower_threshold <= ratio <= upper_threshold:
            date = parse_datetime(datensatz['Datum'], datensatz['Anfang']).date()
            dates_with_ee_ratio.add(date)
    
    sorted_dates = sorted(dates_with_ee_ratio)  # Sortieren von alt zu neu
    
    return sorted_dates

lower_threshold = 0.0  
upper_threshold = 0.2  

dates_with_low_ee = find_dates_with_ee_ratio(energie_daten, energie_daten2, lower_threshold, upper_threshold)

if dates_with_low_ee:
    print(f"Daten mit EE-Anteil zwischen {lower_threshold * 100}% und {upper_threshold * 100}% :")
    for date in dates_with_low_ee:
        print(date)
else:
    print(f"Keine DAten mit EE-Anteil zwischen {lower_threshold * 100}% und {upper_threshold * 100}%.")


if input_date:
    selected_date = datetime.datetime.strptime(str(input_date), "%Y-%m-%d").date()

    # Create the figure and axes objects for the first plot
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(hours, consumption_day, label='Consumption')
    ax1.plot(hours, production_day, label='Production (renewable energy)', linewidth=2.5)
 
    ax1.set_xlabel('Time [Hour]')
    ax1.set_ylabel('Power (MWh)')
    ax1.set_title(f'Energy production and consumption for {selected_date.strftime("%d.%m.%Y")}')
    ax1.fill_between(hours, consumption_day)
    ax1.fill_between(hours, production_day)
    ax1.legend()


    # plt.tight_layout()
    ax1.grid(True)
    ax1.set_xticks(range(0, 24))

# Plot 2
fig2, ax2 = plt.subplots(figsize=(10, 4))
x = range(100)
ax2.bar(range(len(counts)), counts)
ax2.set_title('Anzahl der Viertelstunden mit 1-100 % EE-Anteil')
ax2.set_xticks(x[::5])
ax2.set_xticklabels([f'{i}%' for i in range(0, 100, 5)])

# ... Remaining code omitted for brevity ...

if input_date:
    st.pyplot(fig1)
    st.write("##")
    st.write("##")
    st.subheader('Amount of quarter hours with Renewable Energy in Percent')
    st.markdown("---")
    st.pyplot(fig2)




# reading in production data as a dataframe
df_p = pd.read_csv(csv_datei1, delimiter=";")
df_p['Datum'] = pd.to_datetime(df_p['Datum'], format='%d.%m.%Y')
df_p['Anfang'] = pd.to_datetime(df_p['Anfang'], format='%H:%M')
df_p['Ende'] = pd.to_datetime(df_p['Ende'], format='%H:%M')


df_p['Wasserkraft [MWh] Originalauflösungen'] = df_p['Wasserkraft [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
df_p['Biomasse [MWh] Originalauflösungen'] = df_p['Biomasse [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
df_p['Wind Offshore [MWh] Originalauflösungen'] = df_p['Wind Offshore [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
df_p['Wind Onshore [MWh] Originalauflösungen'] = df_p['Wind Onshore [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
df_p['Photovoltaik [MWh] Originalauflösungen'] = df_p['Photovoltaik [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)

column_name = 'Sonstige Erneuerbare [MWh] Originalauflösungen'

for idx, value in enumerate(df_p[column_name]):
    try:
        df_p.at[idx, column_name] = float(value.replace(".", "").replace(",", "."))
    except (ValueError, AttributeError):
        df_p.at[idx, column_name] = 0


df_p['Kernenergie [MWh] Originalauflösungen'] = df_p['Kernenergie [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
df_p['Braunkohle [MWh] Originalauflösungen'] = df_p['Braunkohle [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
df_p['Steinkohle [MWh] Originalauflösungen'] = df_p['Steinkohle [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
df_p['Erdgas [MWh] Originalauflösungen'] = df_p['Erdgas [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
df_p['Pumpspeicher [MWh] Originalauflösungen'] = df_p['Pumpspeicher [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
df_p['Sonstige Konventionelle [MWh] Originalauflösungen'] = df_p['Sonstige Konventionelle [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)


# setting the date as an index 
df_p.set_index('Datum', inplace=True)


bio_mean = round(df_p['Biomasse [MWh] Originalauflösungen'].mean(), 2)
water_mean = round(df_p['Wasserkraft [MWh] Originalauflösungen'].mean(), 2)
windoff_mean = round(df_p['Wind Offshore [MWh] Originalauflösungen'].mean(), 2)
windon_mean = round(df_p['Wind Onshore [MWh] Originalauflösungen'].mean(), 2)
pv_mean = round(df_p['Photovoltaik [MWh] Originalauflösungen'].mean(), 2)
other_re_mean = round(df_p['Sonstige Erneuerbare [MWh] Originalauflösungen'].mean(), 2)
nuclear_mean = round(df_p['Kernenergie [MWh] Originalauflösungen'].mean(), 2)
bc_mean = round(df_p['Braunkohle [MWh] Originalauflösungen'].mean(), 2)
sc_mean = round(df_p['Steinkohle [MWh] Originalauflösungen'].mean(), 2)
gas_mean = round(df_p['Erdgas [MWh] Originalauflösungen'].mean(), 2)
ps_mean = round(df_p['Pumpspeicher [MWh] Originalauflösungen'].mean(), 2)
other_conv = round(df_p['Sonstige Konventionelle [MWh] Originalauflösungen'].mean(), 2)

# data for 2020 (via filtering)
# filt_20 = ((df_p['Datum'] >= pd.to_datetime('01.01.2020')) & (df_p['Datum'] < pd.to_datetime('01.01.2021')))
# print(df_p.loc[filt_20])


st.subheader("Daily Average Production")
st.write("In the following you can see the daily average over the last three years for the specific production kind")

col1, col2, col3 = st.columns(3)


with col1:
    st.metric(label="Biomass [MWh]", value=bio_mean)
    st.metric(label="Waterpower [MWh]", value=water_mean)
    st.metric(label="Wind Offshore [MWh]", value=windoff_mean)
    st.metric(label="Wind Onshore [MWh]", value=windon_mean)

with col2:
    st.metric(label="Photovoltaic [MWh]", value=pv_mean)
    st.metric(label="Other Renewable [MWh]", value=other_re_mean)
    st.metric(label="Nuclear [MWh]", value=nuclear_mean)
    st.metric(label="Brown Coal [MWh]", value=bc_mean)

with col3:
    st.metric(label="Hard Coal [MWh]", value=sc_mean)
    st.metric(label="Gas [MWh]", value=gas_mean)
    st.metric(label="Pump storage [MWh]", value=ps_mean)
    st.metric(label="Other Conventional [MWh]", value=other_conv)

# Yearly Metrics
st.divider()

# reading in consumption data as a dataframe

df_c = pd.read_csv(csv_datei2, delimiter=";")
df_c['Datum'] = pd.to_datetime(df_c['Datum'], format='%d.%m.%Y')
df_c['Anfang'] = pd.to_datetime(df_c['Anfang'], format='%H:%M')
df_c['Ende'] = pd.to_datetime(df_c['Ende'], format='%H:%M')

df_c['Gesamt (Netzlast) [MWh] Originalauflösungen'] = df_c['Gesamt (Netzlast) [MWh] Originalauflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)

# setting the date as the index
df_c.set_index('Datum', inplace=True)


# df_p.reset_index(inplace=True)

# renewable energy sum
ree_production_sum = ['Wasserkraft [MWh] Originalauflösungen', 'Biomasse [MWh] Originalauflösungen',
                  'Wind Offshore [MWh] Originalauflösungen', 'Wind Onshore [MWh] Originalauflösungen',
                  'Photovoltaik [MWh] Originalauflösungen', 'Sonstige Erneuerbare [MWh] Originalauflösungen']
df_p['Renewable Energy Sum [MWh]'] = df_p[ree_production_sum].sum(axis=1)


# printing yearly bar charts


st.subheader('Production of Renewable Energy')
st.write('As renewable considered are: waterpower, biomass, wind onshore/offshore, photovoltaics and other renewables')


year_options = [2020, 2021, 2022]
year = st.selectbox('Which year would you like to see?', year_options)

if year == 2020:
    start_date = '2020-01-01'
    end_date = '2020-12-31'
    filtered_df_p = df_p[(df_p.index >= start_date) & (df_p.index <= end_date)]
    filtered_df_c = df_c[(df_c.index >= start_date) & (df_c.index <= end_date)]


if year == 2021:
    start_date = '2021-01-01'
    end_date = '2021-12-31'
    filtered_df_p = df_p[(df_p.index >= start_date) & (df_p.index <= end_date)]
    filtered_df_c = df_c[(df_c.index >= start_date) & (df_c.index <= end_date)]


if year == 2022:
    start_date = '2022-01-01'
    end_date = '2022-12-31'
    filtered_df_p = df_p[(df_p.index >= start_date) & (df_p.index <= end_date)]
    filtered_df_c = df_c[(df_c.index >= start_date) & (df_c.index <= end_date)]



total_production = filtered_df_p['Renewable Energy Sum [MWh]'].sum()
total_biomass = filtered_df_p['Biomasse [MWh] Originalauflösungen'].sum()
total_waterpower = filtered_df_p['Wasserkraft [MWh] Originalauflösungen'].sum()
total_windoff = filtered_df_p['Wind Offshore [MWh] Originalauflösungen'].sum()
total_windon = filtered_df_p['Wind Onshore [MWh] Originalauflösungen'].sum()
total_pv = filtered_df_p['Photovoltaik [MWh] Originalauflösungen'].sum()
total_other_ree = filtered_df_p['Sonstige Erneuerbare [MWh] Originalauflösungen'].sum()
total_consumption = filtered_df_c['Gesamt (Netzlast) [MWh] Originalauflösungen'].sum()

fig = px.bar(filtered_df_p, x=filtered_df_p.index, y='Renewable Energy Sum [MWh]')
fig.update_layout(width=800)
st.plotly_chart(fig)


st.subheader('Metrics for ' + str(year))
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Renewable Energy production [MWh]", value=total_production)
    st.metric(label="Consumption [MWh]", value=total_consumption)
    st.metric(label="Biomass [MWh]", value=total_biomass)
    st.metric(label="Waterpower [MWh]", value=total_waterpower)

with col2:
    st.metric(label="Photovoltaic [MWh]", value=total_pv)
    st.metric(label="Other Renewable [MWh]", value=total_other_ree)
    st.metric(label="Wind Offshore [MWh]", value=total_windoff)
    st.metric(label="Wind Onshore [MWh]", value=total_windon)

# showing the metrics as a bar chart

df = pd.DataFrame({'Metrics': ['Renewable Energy production', 'Consumption', 'Biomass', 'Waterpower',
                               'Photovoltaic', 'Other Renewable', 'Wind Offshore', 'Wind Onshore'],
                   'Values': [total_production, total_consumption, total_biomass, total_waterpower,
                              total_pv, total_other_ree, total_windoff, total_windon]})

fig = px.bar(df, x='Metrics', y='Values',
             labels={'Metrics': 'Metrics', 'Values': 'Values'},
             title='Metrics for ' + str(year)+ ' in [MWh]')

st.plotly_chart(fig)

# Dunkelflauten
# reading in installed power 

df_i = pd.read_csv(csv_datei3, delimiter=";")
df_i['Datum'] = pd.to_datetime(df_i['Datum'], format='%d.%m.%Y')
df_i['Anfang'] = pd.to_datetime(df_i['Anfang'], format='%H:%M')
df_i['Ende'] = pd.to_datetime(df_i['Ende'], format='%H:%M')

# df_i['Biomasse [MW] Berechnete Auflösungen'] = df_i['Biomasse [MW] Berechnete Auflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
# df_i['Wasserkraft [MW] Berechnete Auflösungen'] = df_i['Wasserkraft [MW] Berechnete Auflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
# df_i['Wind Offshore [MW] Berechnete Auflösungen'] = df_i['Wind Offshore [MW] Berechnete Auflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
# df_i['Wind Onshore [MW] Berechnete Auflösungen'] = df_i['Wind Onshore [MW] Berechnete Auflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
# df_i['Photovoltaik [MW] Berechnete Auflösungen'] = df_i['Photovoltaik [MW] Berechnete Auflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
# 
# column_name = 'Sonstige Erneuerbare [MW] Berechnete Auflösungen'
# 
# for idx, value in enumerate(df_i[column_name]):
#     try:
#         df_i.at[idx, column_name] = float(value.replace(".", "").replace(",", "."))
#     except (ValueError, AttributeError):
#         df_i.at[idx, column_name] = 0
# 
# df_i['Kernenergie [MW] Berechnete Auflösungen'] = df_i['Kernenergie [MW] Berechnete Auflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
# df_i['Braunkohle [MW] Berechnete Auflösungen'] = df_i['Braunkohle [MW] Berechnete Auflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
# df_i['Steinkohle [MW] Berechnete Auflösungen'] = df_i['Steinkohle [MW] Berechnete Auflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
# df_i['Erdgas [MW] Berechnete Auflösungen'] = df_i['Erdgas [MW] Berechnete Auflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
# df_i['Pumpspeicher [MW] Berechnete Auflösungen'] = df_i['Pumpspeicher [MW] Berechnete Auflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)
# df_i['Sonstige Konventionelle [MW] Berechnete Auflösungen'] = df_i['Sonstige Konventionelle [MW] Berechnete Auflösungen'].str.replace(".", "").str.replace(",", ".").astype(float)


#df_p.reset_index(inplace=True)

# counter = 0
# comparison_df = pd.DataFrame()
# comparison_df['Datum'] = df_p['Datum']

# pv_wind_installed_sum = ['Wind Offshore [MW] Berechnete Auflösungen', 'Wind Offshore [MW] Berechnete Auflösungen', 'Photovoltaik [MW] Berechnete Auflösungen']
# df_i['Pv and Wind installed Power [MWh]'] = df_i[pv_wind_installed_sum].sum(axis=1)

# calculating the minimum power production of 2% to not have a "Dunkelflaute"
# c# omparison_df['threshhold pv and wind[MWh]'] = df_i['Pv and Wind installed Power [MWh]'] * 0.2


# pv_wind_production_sum = ['Photovoltaik [MWh] Originalauflösungen'] + ['Wind Offshore [MWh] Originalauflösungen'] + ['Wind Onshore [MWh] Originalauflösungen']
# comparison_df['Pv and Wind Prodcution Power [MWh]'] = df_p[pv_wind_production_sum].sum(axis=1)

# Collect dates where production is less than the threshold
# less_than_threshold_dates = comparison_df[comparison_df['Pv and Wind Prodcution Power [MWh]'] < comparison_df['threshhold pv and wind[MWh]']]['Datum']

# Increment the counter for each date
# counter += len(less_than_threshold_dates)

# Store the dates in a new column in comparison_df
# comparison_df['Dates Less than Threshold'] = less_than_threshold_dates

# Print the updated comparison_df
# print(comparison_df[['Datum', 'Pv and Wind Prodcution Power [MWh]', 'threshhold pv and wind[MWh]', 'Dates Less than Threshold']])

# Print the counter
# print("Number of dates with production less than the threshold:", counter)

endzeit = time.time()
dauer = endzeit - startzeit
st.write(f"Dauer des Programms: {dauer} Sekunden")