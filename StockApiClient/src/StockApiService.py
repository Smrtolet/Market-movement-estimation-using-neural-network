from datetime import datetime
from iexfinance.stocks import get_historical_data, get_historical_intraday, get_todays_earnings



if __name__ == '__main__':
    start = datetime(2017, 1, 1)
    end = datetime(2018, 1, 1)

    df = get_historical_data("TSLA", start, end, output_format='pandas')
    #n = get_historical_intraday("TSLA", output_format='pandas')
    #print(get_todays_earnings())
    print(df)