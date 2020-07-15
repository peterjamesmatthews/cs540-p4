import csv
import re
import requests as r
import numpy as np
import pandas as pd

globalDeaths = r.get("https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
globalDeaths.raise_for_status()
globalDeaths = re.sub("\"(.+), (.+)\"", r'\g<2> \g<1>', globalDeaths.text)
usDeaths = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv")
DEBUG = False

def load_dataset(data: str) -> np.array:
    # fix "Korea, South" and other one
    raw = re.sub("\"(.+), (.+)\"", r'\g<2> \g<1>', data)
    records = raw.split('\n')
    c_data = dict()
    for record in records[1:-1]:
        record = record.split(',')
        country = record[1]
        deaths = record[-1]
        if country not in c_data.keys():
            c_data[country] = int(deaths)
        else:
            c_data[country] += int(deaths)

    return c_data


def question1() -> str:
    # fix "Korea, South" and other one
    records = globalDeaths.split('\n')
    canada = None
    for record in records[1:-1]:
        record = record.split(',')
        if record[1] == 'Canada':
            if canada is None:
                canada = np.array(record[4:], dtype=int)
            else:
                canada += np.array(record[4:], dtype=int)

    us = np.zeros((1, len(canada)), dtype=int)
    for i, row in usDeaths.iterrows():
        us += (row.iloc[12:]).values.astype(int)

    ret = ''
    for day in canada[:-1]:
        ret += f'{day},'
    ret += f'{canada[-1]}\n'
    for day in us[0][:-1]:
        ret += f'{day},'
    ret += f'{us[0][-1]}'
    return ret


def question2() -> str:
    # fix "Korea, South" and other one
    records = globalDeaths.split('\n')
    canada = None
    for record in records[1:-1]:
        record = record.split(',')
        if record[1] == 'Canada':
            if canada is None:
                canada = np.array(record[4:], dtype=int)
            else:
                canada += np.array(record[4:], dtype=int)
    us = np.zeros((1, len(canada)), dtype=int)
    for i, row in usDeaths.iterrows():
        us += (row.iloc[12:]).values.astype(int)
    us = us[0]
    assert len(us) == len(canada), f'len(us) = {len(us)}\tlen(canada) = {len(canada)}'

    us_diff = np.zeros((1, len(us)-1), dtype=int)
    canada_diff = np.zeros((1, len(canada)-1), dtype=int)
    for i in range(len(us_diff[0])):
        us_diff[0][i] = us[i+1] - us[i]
        canada_diff[0][i] = canada[i+1] - canada[i]

    ret = ''
    for diff in canada_diff[0][:-1]:
        ret += f'{diff},'
    ret += f'{canada_diff[0][-1]}\n'
    for diff in us_diff[0][:-1]:
        ret += f'{diff},'
    ret += f'{us_diff[0][-1]}'
    return ret


def main():
    ds = load_dataset(globalDeaths)

    if DEBUG:
        return 1
    with open('P4.txt', 'w') as f:
        f.write('Outputs:\n')
        f.write('@id\n')
        f.write('pjmatthews\n')
        f.write('@original\n')
        f.write(f'{question1()}\n')
        f.write('@difference\n')
        f.write(f'{question2()}\n')
        f.write('@answer_3\n')
        f.write(f'\n')
        f.write('@parameters\n')
        f.write(f'\n')
        f.write(f'@hacs\n')
        f.write(f'\n')
        f.write('@hacc\n')
        f.write(f'\n')
        f.write('@kmeans\n')
        f.write(f'\n')
        f.write('@centers\n')
        f.write(f'\n')
        f.write('@answer_9\n')
        f.write(f'\n')
        f.write('@answer_10\n')
        f.write('None\n')
    pass


if __name__ == "__main__":
    main()
