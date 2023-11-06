import csv
import matplotlib.pyplot as plt
from datetime import datetime
import time

input_date = input("DD.MM.YYYY: ")

def parse_datetime(date_str, time_str):
    return datetime.strptime(f"{date_str} {time_str}", '%d.%m.%Y %H:%M')

startzeit = time.time()

csv_datei = 'Realisierte_Erzeugung_202001010000_202212312359_Viertelstunde.csv'
csv_datei2 = 'Realisierter_Stromverbrauch_202001010000_202212312359_Viertelstunde.csv'


energie_daten = []
energie_daten2 = []
production = []
consumption = []

with open(csv_datei, 'r') as file:
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
    

selected_date = datetime.strptime(input_date, '%d.%m.%Y')
filtered_data = [datensatz for datensatz in energie_daten if parse_datetime(datensatz['Datum'], datensatz['Anfang']).date() == selected_date.date()]
filtered_data2 = [datensatz1 for datensatz1 in energie_daten2 if parse_datetime(datensatz1['Datum'], datensatz1['Anfang']).date() == selected_date.date()]


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
    year = datetime.strptime(datum, '%d.%m.%Y').year
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
    year = datetime.strptime(datum, '%d.%m.%Y').year
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


plt.figure(figsize=(12, 6))
plt.plot(hours, production_day, label='Production (renewable energy)')
plt.plot(hours, consumption_day, label='Consumption')
plt.xlabel('Time [Hour]')
plt.ylabel('Power (MWh)')
plt.title(f'Energy production and consumption for {selected_date.strftime("%d.%m.%Y")}')
plt.legend()
plt.grid(True)


x = range(100)
plt.figure(figsize=(6, 4)) 
plt.bar(range(len(counts)), counts)
plt.title('Anzahl der Viertelstunden mit 1-100 % EE-Anteil')
plt.xticks(x[::5], [f'{i}%' for i in range(0, 100, 5)])

endzeit = time.time()
dauer = endzeit - startzeit
print()
print()
print(f"Dauer des Programms: {dauer} Sekunden")

plt.show()

plt.show()