from datetime import datetime
from iexfinance.stocks import get_historical_data


if __name__ == '__main__':
    start = datetime(2017, 1, 1)
    end = datetime(2018, 1, 1)

    df = get_historical_data("TSLA", start, end)
    for key in sorted(df.keys()):
        print(key,": ", df[key])