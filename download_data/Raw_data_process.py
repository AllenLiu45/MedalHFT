import os
import zipfile
from datetime import datetime
import re
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import math
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression


# Extract zip files
def extract_and_rename_zip_files(zip_folder_path, extract_to_folder):

    if not os.path.exists(extract_to_folder):
        os.makedirs(extract_to_folder)

    for filename in os.listdir(zip_folder_path):
        if filename.endswith('.zip'):

            zip_file_path = os.path.join(zip_folder_path, filename)


            match = re.match(r'.*-(\d{4})-(\d{2})-(\d{2})\.zip$', filename)
            if match:
                year, month, day = match.groups()

                new_filename = f"{year}-{month}-{day}.csv"
                new_file_path = os.path.join(extract_to_folder, new_filename)

                if os.path.exists(new_file_path):
                    print(f"file {new_filename} already exists, skipping: {filename}")
                    continue


                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:

                    csv_filename = zip_ref.namelist()[0]


                    zip_ref.extract(csv_filename, extract_to_folder)


                    extracted_file_path = os.path.join(extract_to_folder, csv_filename)

                    if not os.path.exists(new_file_path):
                        os.rename(extracted_file_path, new_file_path)
                        print(f"file {filename} -> {new_filename} extracted and renamed")
                    else:

                        os.remove(extracted_file_path)
                        print(f"file {new_filename} already exists, temporary file deleted, skipping: {filename}")

            else:
                print(f"unable to parse file name format: {filename}")


def transform_time(file_folder):
    for filename in os.listdir(file_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(file_folder, filename)
            data = pd.read_csv(file_path)

            if 'open_time' in data.columns:
                data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')

            if 'close_time' in data.columns:
                data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')


            data.to_csv(file_path, index=False)

            print(f"Converted {filename}")

def check_date(file_folder):

    start_date = datetime.strptime('2023-01-01', '%Y-%m-%d')
    end_date = datetime.strptime('2025-09-14', '%Y-%m-%d')


    existing_dates = set()

    for filename in os.listdir(file_folder):
        if filename.endswith('.csv'):

            match = re.match(r'(\d{4})-(\d{2})-(\d{2})\.csv$', filename)
            if match:
                year, month, day = match.groups()
                date_str = f"{year}-{month}-{day}"
                existing_dates.add(datetime.strptime(date_str, '%Y-%m-%d'))


    all_dates = set()
    current_date = start_date
    while current_date <= end_date:
        all_dates.add(current_date)
        current_date += timedelta(days=1)


    missing_dates = sorted(all_dates - existing_dates)


    if missing_dates:
        print("Missing dates:")
        for date in missing_dates:
            print(date.strftime('%Y-%m-%d'))
    else:
        print("No missing dates in the specified range.")


def keep_second_per_minute(file_folder, save_folder):
    for filename in os.listdir(file_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(file_folder, filename)
            data = pd.read_csv(file_path)


            data['timestamp'] = pd.to_datetime(data['timestamp'])


            data['minute'] = data['timestamp'].dt.floor('min')


            grouped = data.groupby('minute')

            def select_last_complete_record(group):

                group_sorted = group.sort_values('timestamp')

                last_timestamp = group_sorted['timestamp'].max()

                last_records = group_sorted[group_sorted['timestamp'] == last_timestamp]

                return last_records


            result_data = grouped.apply(select_last_complete_record, include_groups=False)

            if isinstance(result_data.index, pd.MultiIndex):
                result_data = result_data.reset_index(drop=True)
            else:
                result_data = result_data.reset_index()

            if 'minute' in result_data.columns:
                result_data.drop(columns=['minute'], inplace=True)

            if 'level_0' in result_data.columns:
                result_data.drop(columns=['level_0'], inplace=True)

            if 'minute' in result_data.columns:
                result_data.drop(columns=['minute'], inplace=True)

            output_file_path = os.path.join(save_folder, f"{filename}")
            result_data.to_csv(output_file_path, index=False)

            # print(f"Processed {filename} saved")


def long_orderbook_to_wide(file_folder, save_folder):

    for filename in os.listdir(file_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(file_folder, filename)

            df = pd.read_csv(file_path)
            minute_col = 'timestamp'
            pct_col = 'percentage'
            size_col = 'depth'
            value_col = 'notional'
            nth_per_minute = 0
            fillna = 0.0
            tz = None

            df = df.copy()

            df[minute_col] = pd.to_datetime(df[minute_col])
            if tz is not None:
                if df[minute_col].dt.tz is None:
                    df[minute_col] = df[minute_col].dt.tz_localize('UTC').dt.tz_convert(tz)
                else:
                    df[minute_col] = df[minute_col].dt.tz_convert(tz)
            df['minute'] = df[minute_col].dt.floor('min')

            def side_level(p):
                p = int(p)
                if p < 0:
                    return f"bid{abs(p)}"
                elif p > 0:
                    return f"ask{p}"
                else:

                    return "mid0"
            df['side_level'] = df[pct_col].apply(side_level)


            df = df.sort_values([ 'minute', pct_col, minute_col ])

            df_pick = df.groupby(['minute', pct_col], as_index=False).nth(nth_per_minute).reset_index(drop=True)

            df_pick['size_col_name']  = df_pick['side_level'] + '_size'
            df_pick['value_col_name'] = df_pick['side_level'] + '_value'

            size_wide = df_pick.pivot_table(index='minute',
                                            columns='size_col_name',
                                            values=size_col,
                                            aggfunc='first')
            value_wide = df_pick.pivot_table(index='minute',
                                             columns='value_col_name',
                                             values=value_col,
                                             aggfunc='first')

            wide = pd.concat([size_wide, value_wide], axis=1)

            def sort_key(col):

                if col.startswith('bid'):
                    num = int(col.split('_')[0][3:])
                    return (0, -num, col.endswith('_value'))
                elif col.startswith('ask'):
                    num = int(col.split('_')[0][3:])  # ask{num}
                    return (1, num, col.endswith('_value'))
                else:
                    return (2, 0, 0)
            cols_sorted = sorted(wide.columns, key=sort_key)
            wide = wide.reindex(columns=cols_sorted)

            if fillna is not None:
                wide = wide.fillna(fillna)

            wide = wide.reset_index().rename(columns={'minute': 'timestamp'})

            save_path = os.path.join(save_folder, filename)
            wide.to_csv(save_path, index=False)
            # print(f"Processed {filename} saved")


def add_price(file_folder, save_folder):
    for filename in os.listdir(file_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(file_folder, filename)
            df = pd.read_csv(file_path)

            bid_columns = [col for col in df.columns if col.startswith('bid') and col.endswith('_size')]
            ask_columns = [col for col in df.columns if col.startswith('ask') and col.endswith('_size')]

            for bid_col in sorted(bid_columns, reverse=True):  # bid5, bid4, ..., bid1
                level = bid_col.replace('_size', '')  # 'bid5'
                size_col = f'{level}_size'
                value_col = f'{level}_value'
                price_col = f'{level}_price'

                # price = value/size
                df[price_col] = df[value_col] / df[size_col]

                size_col_index = df.columns.get_loc(size_col)
                price_col_temp = df.pop(price_col)
                df.insert(size_col_index, price_col, price_col_temp)

            for ask_col in sorted(ask_columns):  # ask1, ask2, ..., ask5
                level = ask_col.replace('_size', '')  # 'ask1'
                size_col = f'{level}_size'
                value_col = f'{level}_value'
                price_col = f'{level}_price'

                # price = value/siz
                df[price_col] = df[value_col] / df[size_col]

                size_col_index = df.columns.get_loc(size_col)
                price_col_temp = df.pop(price_col)
                df.insert(size_col_index, price_col, price_col_temp)

            output_file_path = os.path.join(save_folder, filename)
            df.to_csv(output_file_path, index=False)
            # print(f"Processed {filename} saved")


def add_technical_indicators(file_folder, save_folder):
    """
    factors：
    - volume: sum of size
    - ask{n}_size_n:
    - bid{n}_size_n:
    - wap_1: (ask1_price*ask1_size + bid1_price*bid1_size)/(ask1_size+bid1_size)
    - wap_2: (ask2_price*ask2_size + bid2_price*bid2_size)/(ask2_size+bid2_size)
    - wap_balance: |wap_1 - wap_2|
    - buy_spread: |bid1_price - bid5_price|
    - sell_spread: |ask1_price - ask5_price|
    - buy_volume: sum of buy_size
    - sell_volume: sum of sell_size
    - volume_imbalance: (buy_volume-sell_volume)/(buy_volume+sell_volume)
    - price_spread: 2*(ask1_price-bid1_price)/(ask1_price+bid1_price)
    - sell_vwap: Σ(ask{i}_size_n * ask{i}_price) for i=1 to 5
    - buy_vwap: Σ(bid{i}_size_n * bid{i}_price) for i=1 to 5
    ...
    """
    for filename in os.listdir(file_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(file_folder, filename)
            df = pd.read_csv(file_path)

            bid_columns = [col for col in df.columns if col.startswith('bid') and col.endswith('_size')]
            ask_columns = [col for col in df.columns if col.startswith('ask') and col.endswith('_size')]

            size_columns = [col for col in df.columns if col.endswith('_size')]
            df['volume'] = df[size_columns].sum(axis=1)

            for col in size_columns:

                new_col_name = col.replace('_size', '_size_n')

                df[new_col_name] = df[col] / df['volume']

                value_col_name = col.replace('_size', '_value')
                df.drop(columns=[value_col_name], inplace=True)

            # wap_1 ((ask1_price*ask1_size + bid1_price*bid1_size)/(ask1_size+bid1_size))
            if 'ask1_price' in df.columns and 'bid1_price' in df.columns:
                df['wap_1'] = (df['ask1_size'] * df['bid1_price'] + df['bid1_size'] * df['ask1_price']) / \
                              (df['ask1_size'] + df['bid1_size'])

            # wap_2 ((ask2_price*ask2_size + bid2_price*bid2_size)/(ask2_size+bid2_size))
            if 'ask2_price' in df.columns and 'bid2_price' in df.columns:
                df['wap_2'] = (df['ask2_size'] * df['bid2_price'] + df['bid2_size'] * df['ask2_price']) / \
                              (df['ask2_size'] + df['bid2_size'])

            # wap_balance (|wap_1 - wap_2|)
            if 'wap_1' in df.columns and 'wap_2' in df.columns:
                df['wap_balance'] = abs(df['wap_1'] - df['wap_2'])

            # buy_spread (|bid1_price - bid5_price|)
            if 'bid1_price' in df.columns and 'bid5_price' in df.columns:
                df['buy_spread'] = abs(df['bid1_price'] - df['bid5_price'])

            # sell_spread (|ask1_price - ask5_price|)
            if 'ask1_price' in df.columns and 'ask5_price' in df.columns:
                df['sell_spread'] = abs(df['ask1_price'] - df['ask5_price'])

            # buy_volume (sum of bid_size)
            df['buy_volume'] = df[bid_columns].sum(axis=1)

            # sell_volume (sum of ask_size)
            df['sell_volume'] = df[ask_columns].sum(axis=1)

            # volume_imbalance ((buy_volume-sell_volume)/(buy_volume+sell_volume))
            df['volume_imbalance'] = (df['buy_volume'] - df['sell_volume']) / (df['buy_volume'] + df['sell_volume'])

            # price_spread (2*(ask1_price-bid1_price)/(ask1_price+bid1_price))
            if 'ask1_price' in df.columns and 'bid1_price' in df.columns:
                df['price_spread'] = 2 * (df['ask1_price'] - df['bid1_price']) / (df['ask1_price'] + df['bid1_price'])

            # sell_vwap (Σ(ask{i}_size_n * ask{i}_price) for i=1 to 5)
            sell_vwap_components = []
            for i in range(1, 6):
                size_n_col = f'ask{i}_size_n'
                price_col = f'ask{i}_price'
                if size_n_col in df.columns and price_col in df.columns:
                    component = df[size_n_col] * df[price_col]
                    sell_vwap_components.append(component)

            if sell_vwap_components:
                df['sell_vwap'] = sum(sell_vwap_components)

            # buy_vwap (Σ(bid{i}_size_n * bid{i}_price) for i=1 to 5)
            buy_vwap_components = []
            for i in range(1, 6):
                size_n_col = f'bid{i}_size_n'
                price_col = f'bid{i}_price'
                if size_n_col in df.columns and price_col in df.columns:
                    component = df[size_n_col] * df[price_col]
                    buy_vwap_components.append(component)

            if buy_vwap_components:
                df['buy_vwap'] = sum(buy_vwap_components)

            output_file_path = os.path.join(save_folder, filename)
            df.to_csv(output_file_path, index=False)
            print(f"Processed {filename} saved")


def fill_interval(minute_folder, save_folder, interval):

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for filename in os.listdir(minute_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(minute_folder, filename)
            df = pd.read_csv(file_path)

            df['timestamp'] = pd.to_datetime(df['timestamp'])

            for col in df.columns:
                if col != 'timestamp':
                    df.rename(columns={col: f'{interval}min_{col}'}, inplace=True)


            start_time = df['timestamp'].min().floor('min')
            end_time = df['timestamp'].max().ceil('min')

            full_time_range = pd.date_range(start=start_time, end=end_time, freq='1min')

            full_time_df = pd.DataFrame({'timestamp': full_time_range})

            merged_df = pd.merge(full_time_df, df, on='timestamp', how='left')

            merged_df = merged_df.ffill()


            nan_count = merged_df.isnull().sum()
            cols_with_nan = nan_count[nan_count > 0]

            if len(cols_with_nan) > 0:
                print(f"file {filename} exists NaN values:")
                for col, count in cols_with_nan.items():
                    print(f"  {col}: {count} NaN values")

                merged_df = merged_df.ffill().bfill()

                remaining_nan = merged_df.isnull().sum()
                remaining_cols_with_nan = remaining_nan[remaining_nan > 0]

                if len(remaining_cols_with_nan) > 0:
                    print(f"file {filename} still exists NaN values")

            output_file_path = os.path.join(save_folder, filename)
            merged_df.to_csv(output_file_path, index=False)
            print(f"Filled {filename} with {interval}-minute interval data")


def match_1minute_and_orderbook_data(minute_folder, orderbook_folder, save_folder):

    minute_data = []
    for filename in os.listdir(minute_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(minute_folder, filename)
            df = pd.read_csv(file_path)
            minute_data.append(df)


    if minute_data:
        minute_df = pd.concat(minute_data, ignore_index=True)

        minute_df['timestamp'] = pd.to_datetime(minute_df['timestamp'])

        ohlc_columns = ['timestamp', 'open', 'high', 'low', 'close']
        minute_df = minute_df[ohlc_columns]

        for filename in os.listdir(orderbook_folder):
            if filename.endswith('.csv'):
                orderbook_file_path = os.path.join(orderbook_folder, filename)
                orderbook_df = pd.read_csv(orderbook_file_path)

                orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'])

                merged_df = pd.merge(orderbook_df, minute_df, on='timestamp', how='left')

                output_file_path = os.path.join(save_folder, filename)
                merged_df.to_csv(output_file_path, index=False)
                print(f"Merged {filename} saved")


def match_all_minute_and_orderbook_data(minute_folder, three_minute_folder, five_minute_folder, orderbook_folder,
                                        save_folder):

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    minute_data = []
    for filename in os.listdir(minute_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(minute_folder, filename)
            df = pd.read_csv(file_path)
            minute_data.append(df)

    three_minute_data = []
    for filename in os.listdir(three_minute_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(three_minute_folder, filename)
            df = pd.read_csv(file_path)
            three_minute_data.append(df)

    five_minute_data = []
    for filename in os.listdir(five_minute_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(five_minute_folder, filename)
            df = pd.read_csv(file_path)
            five_minute_data.append(df)

    if minute_data:
        minute_df = pd.concat(minute_data, ignore_index=True)

        minute_df['timestamp'] = pd.to_datetime(minute_df['timestamp'])

        columns_to_keep = [col for col in minute_df.columns if col != 'timestamp']
        columns_to_keep.insert(0, 'timestamp')
        minute_df = minute_df[columns_to_keep]

        cols_to_rename = {}
        for col in minute_df.columns:
            if col not in ['timestamp', 'open', 'high', 'low', 'close']:
                cols_to_rename[col] = f'1min_{col}'

        minute_df.rename(columns=cols_to_rename, inplace=True)

    if three_minute_data:
        three_minute_df = pd.concat(three_minute_data, ignore_index=True)

        three_minute_df['timestamp'] = pd.to_datetime(three_minute_df['timestamp'])

        columns_to_keep = [col for col in three_minute_df.columns if col != 'timestamp']
        columns_to_keep.insert(0, 'timestamp')
        three_minute_df = three_minute_df[columns_to_keep]

    if five_minute_data:
        five_minute_df = pd.concat(five_minute_data, ignore_index=True)

        five_minute_df['timestamp'] = pd.to_datetime(five_minute_df['timestamp'])

        columns_to_keep = [col for col in five_minute_df.columns if col != 'timestamp']
        columns_to_keep.insert(0, 'timestamp')
        five_minute_df = five_minute_df[columns_to_keep]


    for filename in os.listdir(orderbook_folder):
        if filename.endswith('.csv'):
            orderbook_file_path = os.path.join(orderbook_folder, filename)
            orderbook_df = pd.read_csv(orderbook_file_path)

            orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'])

            merged_df = pd.merge(orderbook_df, minute_df, on='timestamp', how='left')

            if three_minute_data:
                merged_df = pd.merge(merged_df, three_minute_df, on='timestamp', how='left')

            if five_minute_data:
                merged_df = pd.merge(merged_df, five_minute_df, on='timestamp', how='left')

            output_file_path = os.path.join(save_folder, filename)
            merged_df.to_csv(output_file_path, index=False)
            print(f"Merged {filename} saved")


def process_orderbook_data(raw_folder, save_folder):

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    print("Only keep the last record of each minute")
    keep_second_per_minute(raw_folder, save_folder)

    print("Processing OrderBook data")
    long_orderbook_to_wide(save_folder, save_folder)

    print("Adding price column to OrderBook data")
    add_price(save_folder, save_folder)

    print("Adding technical indicators to OrderBook data")
    add_technical_indicators(save_folder, save_folder)


def process_minute_data(raw_folder, save_folder, interval):

    for filename in os.listdir(raw_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(raw_folder, filename)
            df = pd.read_csv(file_path)

            if 'open_time' in df.columns:
                df['open_time'] = pd.to_datetime(df['open_time'])

                if interval == '1minute':
                    df['timestamp'] = df['open_time'] + pd.Timedelta(minutes=1)
                    df['taker_buy_volume_ratio'] = df['taker_buy_volume'] / df['volume']
                    df['taker_buy_quote_volume_ratio'] = df['taker_buy_quote_volume'] / df['quote_volume']
                    df['order_flow_imbalance_ratio'] = (df['taker_buy_volume'] - (
                                df['volume'] - df['taker_buy_volume'])) / df['volume']
                    df['kmid'] = df['close'] - df['open']
                    df['kmid2'] = (df['close'] - df['open']) / (df['high'] - df['low'])
                    df['klen'] = df['high'] - df['low']
                    df['kup'] = df['high'] - np.maximum(df['open'], df['close'])
                    df['kup2'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'])
                    df['klow'] = (np.minimum(df['open'], df['close']) - df['low'])
                    df['klow2'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['high'] - df['low'])
                    df['ksft'] = 2 * df['close'] - df['high'] - df['low']
                    df['ksft2'] = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'])
                elif interval == '3minute':
                    df['timestamp'] = df['open_time'] + pd.Timedelta(minutes=3)
                    df['taker_buy_volume_ratio'] = df['taker_buy_volume']/df['volume']
                    df['taker_buy_quote_volume_ratio'] = df['taker_buy_quote_volume']/df['quote_volume']
                    df['order_flow_imbalance_ratio'] = (df['taker_buy_volume'] - (df['volume'] - df['taker_buy_volume'])) / df['volume']
                    df['kmid'] = df['close'] - df['open']
                    df['kmid2'] = (df['close'] - df['open']) / (
                                df['high'] - df['low'])
                    df['klen'] = df['high'] - df['low']
                    df['kup'] = df['high'] - np.maximum(df['open'], df['close'])
                    df['kup2'] = (df['high'] - np.maximum(df['open'], df['close'])) / (
                                df['high'] - df['low'])
                    df['klow'] = (np.minimum(df['open'], df['close']) - df['low'])
                    df['klow2'] = (np.minimum(df['open'], df['close']) - df['low']) / (
                                df['high'] - df['low'])
                    df['ksft'] = 2 * df['close'] - df['high'] - df['low']
                    df['ksft2'] = (2 * df['close'] - df['high'] - df['low']) / (
                                df['high'] - df['low'])
                elif interval == '5minute':
                    df['timestamp'] = df['open_time'] + pd.Timedelta(minutes=5)
                    df['taker_buy_volume_ratio'] = df['taker_buy_volume']/df['volume']
                    df['taker_buy_quote_volume_ratio'] = df['taker_buy_quote_volume']/df['quote_volume']
                    df['order_flow_imbalance_ratio'] = (df['taker_buy_volume'] - (df['volume'] - df['taker_buy_volume'])) / df['volume']
                    df['kmid'] = df['close'] - df['open']
                    df['kmid2'] = (df['close'] - df['open']) / (
                                df['high'] - df['low'])
                    df['klen'] = df['high'] - df['low']
                    df['kup'] = df['high'] - np.maximum(df['open'], df['close'])
                    df['kup2'] = (df['high'] - np.maximum(df['open'], df['close'])) / (
                                df['high'] - df['low'])
                    df['klow'] = (np.minimum(df['open'], df['close']) - df['low'])
                    df['klow2'] = (np.minimum(df['open'], df['close']) - df['low']) / (
                                df['high'] - df['low'])
                    df['ksft'] = 2 * df['close'] - df['high'] - df['low']
                    df['ksft2'] = (2 * df['close'] - df['high'] - df['low']) / (
                                df['high'] - df['low'])
                else:
                    raise ValueError("interval parameter must be one of '1minute', '3minute', or '5minute")
            df.drop(columns=['open_time', 'close_time', 'ignore', 'count'], inplace=True)

            if 'timestamp' in df.columns:
                timestamp_col = df.pop('timestamp')
                df.insert(0, 'timestamp', timestamp_col)

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            output_file_path = os.path.join(save_folder, filename)
            df.to_csv(output_file_path, index=False)
            print(f"Processed {filename} with {interval} interval")


def add_trend_features_and_convert_to_feather(raw_folder, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    def get_slope_window(window):
        N, Wn = 1, 0.05
        b, a = butter(N, Wn, btype='low')
        y = filtfilt(b, a, window.values)
        X = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        return model.coef_[0]

    csv_files = [f for f in os.listdir(raw_folder) if f.endswith('.csv')]
    csv_files.sort()

    dataframes = []
    for filename in csv_files:
        file_path = os.path.join(raw_folder, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)

        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])

        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)


        if 'bid1_price' in combined_df.columns:
            combined_df['log_return_bid1_price'] = np.log(
                combined_df['bid1_price'] / combined_df['bid1_price'].shift(1)).bfill() * 10000

        if 'bid2_price' in combined_df.columns:
            combined_df['log_return_bid2_price'] = np.log(
                combined_df['bid2_price'] / combined_df['bid2_price'].shift(1)).bfill() * 10000

        if 'ask1_price' in combined_df.columns:
            combined_df['log_return_ask1_price'] = np.log(
                combined_df['ask1_price'] / combined_df['ask1_price'].shift(1)).bfill() * 10000

        if 'ask2_price' in combined_df.columns:
            combined_df['log_return_ask2_price'] = np.log(
                combined_df['ask2_price'] / combined_df['ask2_price'].shift(1)).bfill() * 10000

        if 'wap_1' in combined_df.columns:
            combined_df['log_return_wap_1'] = np.log(combined_df['wap_1'] / combined_df['wap_1'].shift(1)).bfill() * 10000

        if 'wap_2' in combined_df.columns:
            combined_df['log_return_wap_2'] = np.log(combined_df['wap_2'] / combined_df['wap_2'].shift(1)).bfill() * 10000

        combined_df['kmid'] = combined_df['close'] - combined_df['open']
        combined_df['kmid2'] = (combined_df['close'] - combined_df['open'])/ (combined_df['high'] - combined_df['low'])
        combined_df['klen'] = combined_df['high'] - combined_df['low']
        combined_df['kup'] = combined_df['high'] - np.maximum(combined_df['open'], combined_df['close'])
        combined_df['kup2'] = (combined_df['high'] - np.maximum(combined_df['open'], combined_df['close'])) / (combined_df['high'] - combined_df['low'])
        combined_df['klow'] = (np.minimum(combined_df['open'], combined_df['close']) - combined_df['low'])
        combined_df['klow2'] = (np.minimum(combined_df['open'], combined_df['close']) - combined_df['low']) / (combined_df['high'] - combined_df['low'])
        combined_df['ksft'] = 2 * combined_df['close'] - combined_df['high'] - combined_df['low']
        combined_df['ksft2'] = (2 * combined_df['close'] - combined_df['high'] - combined_df['low']) / (combined_df['high'] - combined_df['low'])

        combined_df['ask1_price_trend_60'] = (combined_df['ask1_price'] - combined_df['ask1_price'].rolling(window=60).mean()) / combined_df['ask1_price'].rolling(window=60).std()
        combined_df['bid1_price_trend_60'] = (combined_df['bid1_price'] - combined_df['bid1_price'].rolling(window=60).mean()) / combined_df['bid1_price'].rolling(window=60).std()
        combined_df['buy_spread_trend_60'] = (combined_df['buy_spread'] - combined_df['buy_spread'].rolling(window=60).mean()) / combined_df['buy_spread'].rolling(window=60).std()
        combined_df['sell_spread_trend_60'] = (combined_df['sell_spread'] - combined_df['sell_spread'].rolling(window=60).mean()) / combined_df['sell_spread'].rolling(window=60).std()
        combined_df['wap_1_trend_60'] = (combined_df['wap_1'] - combined_df['wap_1'].rolling(window=60).mean()) / combined_df['wap_1'].rolling(window=60).std()
        combined_df['wap_2_trend_60'] = (combined_df['wap_2'] - combined_df['wap_2'].rolling(window=60).mean()) / combined_df['wap_2'].rolling(window=60).std()
        combined_df['buy_vwap_trend_60'] = (combined_df['buy_vwap'] - combined_df['buy_vwap'].rolling(window=60).mean()) / combined_df['buy_vwap'].rolling(window=60).std()
        combined_df['sell_vwap_trend_60'] = (combined_df['sell_vwap'] - combined_df['sell_vwap'].rolling(window=60).mean()) / combined_df['sell_vwap'].rolling(window=60).std()
        combined_df['volume_trend_60'] = (combined_df['volume'] - combined_df['volume'].rolling(window=60).mean()) / combined_df['volume'].rolling(window=60).std()

        combined_df['trend_360'] = combined_df['close'].rolling(window=360).apply(get_slope_window)
        combined_df['return'] = combined_df['close'].pct_change(fill_method=None).fillna(0)

        combined_df['vol_360'] = combined_df['return'].rolling(window=360).std() * 100
        combined_df['liq_360'] = combined_df['volume'] / combined_df['volume'].rolling(window=360).mean()

        combined_df = combined_df.iloc[360:].reset_index(drop=True)

        combined_df = combined_df.ffill().bfill()
        print(f"fill nan values: {combined_df.isnull().sum().sum()}")
        print(f"fill na values: {combined_df.isna().sum().sum()}")

        # feather_file_path = os.path.join(save_folder, 'df_whole.feather')
        # combined_df.to_feather(feather_file_path)
        # combined_df.to_csv('df_whole.csv', index=False)

        total_rows = len(combined_df)
        train_end = int(total_rows * 0.7)
        val_end = int(total_rows * 0.85)

        train_df = combined_df.iloc[:train_end]
        train_df = train_df[(train_df['timestamp'] < '2024-11-25 4:45:00') | (train_df['timestamp'] >= '2023-01-01 6:15:00')]

        val_df = combined_df.iloc[train_end:val_end]

        val_df = val_df[(val_df['timestamp'] < '2025-04-16 10:31:00') | (val_df['timestamp'] > '2025-05-19 12:12:00')]
        test_df = combined_df.iloc[val_end:]

        test_df = test_df[(test_df['timestamp'] < '2025-04-16 10:31:00') | (test_df['timestamp'] >= '2025-05-19 12:15:00')]

        train_df.to_feather(os.path.join(save_folder, 'df_train.feather'))
        val_df.to_feather(os.path.join(save_folder, 'df_val.feather'))
        test_df.to_feather(os.path.join(save_folder, 'df_test.feather'))

        train_df.to_csv(os.path.join(save_folder, 'df_train.csv'), index=False)
        val_df.to_csv(os.path.join(save_folder, 'df_val.csv'), index=False)
        test_df.to_csv(os.path.join(save_folder, 'df_test.csv'), index=False)

        df_whole = pd.concat([train_df, val_df, test_df], ignore_index=True)
        df_whole.to_csv(os.path.join(save_folder, 'df_whole.csv'), index=False)
        df_whole.to_feather(os.path.join(save_folder, 'df_whole.feather'))

        print(f"Combined data saved")


def check_nan_and_fill_na(file_path):

    for filename in os.listdir(file_path):
        if filename.endswith('.csv'):
            csv_file_path = os.path.join(file_path, filename)

            try:
                # 读取CSV文件
                df = pd.read_csv(csv_file_path)

                # 检查NaN和空值
                nan_count = df.isnull().sum()
                empty_count = (df == '').sum()

                print(f"\nfile filter: {filename}")
                print(f"rows count: {len(df)}")
                print("NaN value:")
                for col, count in nan_count.items():
                    if count > 0:
                        print(f"  {col}: {count}")

                print("empty value:")
                for col, count in empty_count.items():
                    if count > 0:
                        print(f"  {col}: {count}")

                df = df.replace('', pd.NA)

                df_filled = df.ffill().bfill()

                for col in df_filled.columns:
                    if df_filled[col].isnull().any():
                        if df_filled[col].dtype in ['int64', 'float64']:
                            df_filled[col] = df_filled[col].fillna(0)
                        else:
                            df_filled[col] = df_filled[col].fillna('')

                df_filled.to_csv(csv_file_path, index=False)

                print(f"saved file: {filename}")

            except Exception as e:
                print(f"handle {filename} error: {e}")


if __name__ == '__main__':

    cryptocurrency = 'LINKUSDT'

    # step1 unzip
    # 1-min
    zip_folder_path = f''
    extract_to_1min_folder = f''
    extract_and_rename_zip_files(zip_folder_path, extract_to_1min_folder)
    # 3-min
    zip_folder_path = f''
    extract_to_3min_folder = f''
    extract_and_rename_zip_files(zip_folder_path, extract_to_3min_folder)
    # 5-min
    zip_folder_path = f''
    extract_to_5min_folder = f''
    extract_and_rename_zip_files(zip_folder_path, extract_to_5min_folder)
    # OrderBook
    zip_folder_path = f''
    extract_to_orderbook_folder = f''
    extract_and_rename_zip_files(zip_folder_path, extract_to_orderbook_folder)

    # step2 transform time

    transform_time(extract_to_1min_folder)
    transform_time(extract_to_3min_folder)
    transform_time(extract_to_5min_folder)

    # check date
    check_date(extract_to_1min_folder)
    check_date(extract_to_3min_folder)
    check_date(extract_to_5min_folder)

    #step 3
    # process minute data
    process_1min_folder = f''
    process_3min_folder = f''
    process_5min_folder = f''

    process_minute_data(extract_to_1min_folder, process_1min_folder, '1minute')
    process_minute_data(extract_to_3min_folder, process_3min_folder, '3minute')
    process_minute_data(extract_to_5min_folder, process_5min_folder, '5minute')

    #step 4
    # process orderBook data
    process_orderbook_folder = f''
    process_orderbook_data(extract_to_orderbook_folder, process_orderbook_folder)

    #step 5
    # fill data
    fill_3min_folder = f''
    fill_5min_folder = f''
    fill_interval(process_1min_folder, fill_3min_folder, 3)
    fill_interval(process_5min_folder, fill_5min_folder, 5)

    # step 6
    # match minute data and OrderBook data
    combine_folder = f''
    match_all_minute_and_orderbook_data(process_3min_folder, fill_3min_folder, fill_5min_folder, process_orderbook_folder, combine_folder)

    # step 7
    # check NaN and fill na
    check_nan_and_fill_na(combine_folder)

    # step 8
    # convert to feather format
    final_folder = f''
    add_trend_features_and_convert_to_feather(combine_folder, final_folder)

