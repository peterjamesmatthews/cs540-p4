import requests as r
import numpy as np
import pandas as pd

URL = "https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"


def load_dataset(url: str) -> np.array:
    df = pd.read_csv(url)

    for i, row in df.iterrows():
        if row['Province/State'] != float('nan'):
            print(i)

    return df
    return df.values, df.columns


def main():
    ds = load_dataset(URL)
    print(ds)

    pass


if __name__ == "__main__":
    main()
