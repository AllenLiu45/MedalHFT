import numpy as np
import pandas as pd
import os
import pickle
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LinearRegression
import argparse
import matplotlib.pyplot as plt
from ruptures import Pelt
import matplotlib.ticker as ticker
from scipy.signal import butter, filtfilt
import shutil
import concurrent.futures


def merge_trend_points(change_points, prices, avg_threshold, slope_threshold):
    if len(change_points) <= 1:

        if len(change_points) == 1:
            return [0, change_points[0]] if change_points[0] > 0 else [change_points[0], len(prices)]
        else:
            return [0, len(prices)] if len(prices) > 0 else [0, 1]

    if len(change_points) == 2:
        return [change_points[0], change_points[1]]

    merged = [change_points[0]]

    for i in range(1, len(change_points) - 1):

        prev_segment = prices[merged[-1]:change_points[i]]

        current_segment = prices[change_points[i]:change_points[i + 1]]

        should_merge = False

        if len(prev_segment) > 0 and len(current_segment) > 0:
            mean_diff_small = np.abs(np.mean(current_segment) - np.mean(prev_segment)) < avg_threshold

            prev_trend = "UP" if prev_segment[-1] > prev_segment[0] else "DOWN"
            current_trend = "UP" if current_segment[-1] > current_segment[0] else "DOWN"
            trend_same = prev_trend == current_trend

            prev_slope = (prev_segment[-1] - prev_segment[0]) / max(len(prev_segment), 1)
            current_slope = (current_segment[-1] - current_segment[0]) / max(len(current_segment), 1)
            slope_diff_small = np.abs(prev_slope - current_slope) < slope_threshold

            should_merge = mean_diff_small or trend_same or slope_diff_small

        if not should_merge:
            if change_points[i] != merged[-1]:
                merged.append(change_points[i])

    if len(change_points) > 0 and change_points[-1] not in merged:
        merged.append(change_points[-1])
        if len(merged) > 2:
            prev_segment = prices[merged[-3]:merged[-2]]
            current_segment = prices[merged[-2]:merged[-1]]

            should_merge = False
            if len(prev_segment) > 0 and len(current_segment) > 0:

                mean_diff_small = np.abs(np.mean(current_segment) - np.mean(prev_segment)) < avg_threshold

                prev_trend = "UP" if prev_segment[-1] > prev_segment[0] else "DOWN"
                current_trend = "UP" if current_segment[-1] > current_segment[0] else "DOWN"
                trend_same = prev_trend == current_trend

                prev_slope = (prev_segment[-1] - prev_segment[0]) / max(len(prev_segment), 1)
                current_slope = (current_segment[-1] - current_segment[0]) / max(len(current_segment), 1)
                slope_diff_small = np.abs(prev_slope - current_slope) < slope_threshold

                should_merge = mean_diff_small or trend_same or slope_diff_small
                if should_merge:
                    merged.remove(merged[-2])

    merged = sorted(list(set(merged)))

    return merged


def merge_volatility_points(change_points, prices, threshold):
    if len(change_points) <= 1:

        if len(change_points) == 1:
            return [0, change_points[0]] if change_points[0] > 0 else [change_points[0], len(prices)]
        else:
            return [0, len(prices)] if len(prices) > 0 else [0, 1]

    if len(change_points) == 2:
        return [change_points[0], change_points[1]]

    merged = [change_points[0]]

    for i in range(1, len(change_points) - 1):

        prev_segment = prices[merged[-1]:change_points[i]]

        current_segment = prices[change_points[i]:change_points[i + 1]]

        should_merge = False

        if len(prev_segment) > 0 and len(current_segment) > 0:
            prev_std = np.std(prev_segment)
            current_std = np.std(current_segment)

            std_diff_small = np.abs(prev_std - current_std) < threshold

            should_merge = std_diff_small

        if not should_merge:
            if change_points[i] != merged[-1]:
                merged.append(change_points[i])

    if len(change_points) > 0 and change_points[-1] not in merged:
        merged.append(change_points[-1])
        if len(merged) > 2:
            prev_segment = prices[merged[-3]:merged[-2]]
            current_segment = prices[merged[-2]:merged[-1]]

            should_merge = False
            if len(prev_segment) > 0 and len(current_segment) > 0:
                prev_std = np.std(prev_segment)
                current_std = np.std(current_segment)

                std_diff_small = np.abs(prev_std - current_std) < threshold

                should_merge = std_diff_small
                if should_merge:
                    merged.remove(merged[-2])

    merged = sorted(list(set(merged)))

    return merged


def smooth_data(data, N, Wn):
    Wn = np.clip(Wn, 0.01, 0.4)

    b, a = butter(N, Wn, btype='low')
    # 添加错误处理
    try:
        smoothed = filtfilt(b, a, data)
        return smoothed
    except ValueError as e:
        print(f"ERROR: {e}")
        return data


def label_trend_pelt(df, args, data_type):
    model = args.trend_method
    penalty = args.trend_penalty
    avg_threshold = args.trend_avg_threshold
    slope_threshold = args.trend_slope_threshold
    dataset = args.dataset

    prices = df['close'].values
    prices = pd.to_numeric(prices, errors='coerce')
    prices = prices[~np.isnan(prices)]

    prices_standardized = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))
    prices = prices_standardized
    prices = smooth_data(prices, 5, 0.05)

    chunk_size = 30 * 24 * 60
    all_change_points = []

    for i in range(0, len(prices), chunk_size):
        chunk_prices = prices[i:i + chunk_size]
        prices_2d = chunk_prices.reshape(-1, 1)

        algo = Pelt(model).fit(prices_2d)
        chunk_change_points = algo.predict(pen=penalty)

        adjusted_change_points = [cp + i for cp in chunk_change_points if cp < len(chunk_prices)]
        all_change_points.extend(adjusted_change_points)

    if 0 not in all_change_points:
        all_change_points.append(0)
    if len(prices) not in all_change_points:
        all_change_points.append(len(prices))

    all_change_points = sorted(all_change_points)
    merged_change_points = merge_trend_points(all_change_points, prices, avg_threshold, slope_threshold)

    final_change_points = [merged_change_points[0]]

    for i in range(len(merged_change_points) - 1):
        start_idx = merged_change_points[i]
        end_idx = merged_change_points[i + 1]
        segment = prices[start_idx:end_idx]
        if len(segment) > 0:
            trend = "UP" if segment[-1] > segment[0] else "DOWN"

            if len(final_change_points) >= 2 and i > 0:
                prev_start_idx = final_change_points[-2]
                prev_end_idx = final_change_points[-1]
                prev_segment = prices[prev_start_idx:prev_end_idx]

                if len(prev_segment) > 0:
                    prev_trend = "UP" if prev_segment[-1] > prev_segment[0] else "DOWN"

                    if prev_trend == trend:
                        final_change_points.pop()

            final_change_points.append(end_idx)

    trend_data_dir = f'./MyData/{dataset}/trend_data/{model}/{data_type}'
    os.makedirs(trend_data_dir, exist_ok=True)

    trend_labels = []
    trend_files = []

    for i in range(len(final_change_points) - 1):
        start_idx = final_change_points[i]
        end_idx = final_change_points[i + 1]
        segment = df.iloc[start_idx:end_idx].copy()
        segment = segment.reset_index(drop=True)

        price_segment = prices[start_idx:end_idx]
        trend_label = 1 if price_segment[-1] > price_segment[0] else -1

        filename = f'df_{i}.feather'
        filepath = os.path.join(trend_data_dir, filename)
        segment.to_feather(filepath)

        trend_labels.append(trend_label)
        trend_files.append(filename)

    labels_dict = {-1: [], 1: []}
    for i, label in enumerate(trend_labels):
        labels_dict[label].append(i)

    with open(os.path.join(trend_data_dir, 'trend_labels.pkl'), 'wb') as f:
        pickle.dump(labels_dict, f)

    print(f"\nTrend {data_type}******** Total segments: {len(trend_files)}")


def label_volatility_pelt(df, args, data_type):
    dataset = args.dataset
    model = args.vol_method
    penalty = args.vol_penalty
    threshold = args.vol_threshold

    prices = df['close'].values
    prices = pd.to_numeric(prices, errors='coerce')
    prices = prices[~np.isnan(prices)]

    prices_standardized = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))
    prices = prices_standardized
    prices = smooth_data(prices, 5, 0.05)

    chunk_size = 30 * 24 * 60
    all_change_points = []

    for i in range(0, len(prices), chunk_size):
        chunk_prices = prices[i:i + chunk_size]
        prices_2d = chunk_prices.reshape(-1, 1)

        algo = Pelt(model).fit(prices_2d)
        chunk_change_points = algo.predict(pen=penalty)

        adjusted_change_points = [cp + i for cp in chunk_change_points if cp < len(chunk_prices)]
        all_change_points.extend(adjusted_change_points)

    if 0 not in all_change_points:
        all_change_points.append(0)
    if len(prices) not in all_change_points:
        all_change_points.append(len(prices))

    all_change_points = sorted(all_change_points)
    merged_change_points = merge_volatility_points(all_change_points, prices, threshold=threshold)

    merged_all_std_devs = []
    for i in range(len(merged_change_points) - 1):
        start_idx = merged_change_points[i]
        end_idx = merged_change_points[i + 1]
        segment = prices[start_idx:end_idx]
        if len(segment) > 0:
            merged_all_std_devs.append(np.std(segment))
    merged_mean_std = np.mean(merged_all_std_devs) if merged_all_std_devs else np.std(prices)

    final_change_points = [merged_change_points[0]]

    for i in range(len(merged_change_points) - 1):
        start_idx = merged_change_points[i]
        end_idx = merged_change_points[i + 1]
        segment = prices[start_idx:end_idx]
        if len(segment) > 0:
            std_dev = np.std(segment)
            volatility_type = "High Volatility" if std_dev > merged_mean_std else "Low Volatility"

            if len(final_change_points) >= 2 and i > 0:

                prev_start_idx = final_change_points[-2]
                prev_end_idx = final_change_points[-1]
                prev_segment = prices[prev_start_idx:prev_end_idx]

                if len(prev_segment) > 0:
                    prev_std_dev = np.std(prev_segment)
                    prev_volatility_type = "High Volatility" if prev_std_dev > merged_mean_std else "Low Volatility"

                    if volatility_type == prev_volatility_type:
                        final_change_points.pop()

            final_change_points.append(end_idx)

    vol_data_dir = f'./MyData/{dataset}/vol_data/{model}/{data_type}'
    os.makedirs(vol_data_dir, exist_ok=True)

    vol_labels = []
    vol_files = []

    for i in range(len(final_change_points) - 1):
        start_idx = final_change_points[i]
        end_idx = final_change_points[i + 1]
        segment = df.iloc[start_idx:end_idx].copy()
        segment = segment.reset_index(drop=True)

        price_segment = prices[start_idx:end_idx]
        std_dev = np.std(price_segment)
        vol_label = 1 if std_dev > merged_mean_std else -1

        filename = f'df_{i}.feather'
        filepath = os.path.join(vol_data_dir, filename)
        segment.to_feather(filepath)

        vol_labels.append(vol_label)
        vol_files.append(filename)

    labels_dict = {-1: [], 1: []}
    for i, label in enumerate(vol_labels):
        labels_dict[label].append(i)

    with open(os.path.join(vol_data_dir, 'vol_labels.pkl'), 'wb') as f:
        pickle.dump(labels_dict, f)

    print(f"\nVolatility {data_type}******** Total segments: {len(vol_files)}")


def label_liquidity_pelt(df, args, data_type):
    dataset = args.dataset
    model = args.liq_method
    penalty = args.liq_penalty
    avg_threshold = args.liq_avg_threshold
    slope_threshold = args.liq_slope_threshold

    volume = df['volume'].values
    volume = pd.to_numeric(volume, errors='coerce')
    volume = volume[~np.isnan(volume)]

    volume_standardized = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    volume = volume_standardized
    volume = smooth_data(volume, 5, 0.05)

    chunk_size = 30 * 24 * 60
    all_change_points = []

    for i in range(0, len(volume), chunk_size):
        chunk_volume = volume[i:i + chunk_size]
        volume_2d = chunk_volume.reshape(-1, 1)

        algo = Pelt(model).fit(volume_2d)
        chunk_change_points = algo.predict(pen=penalty)

        adjusted_change_points = [cp + i for cp in chunk_change_points if cp < len(chunk_volume)]
        all_change_points.extend(adjusted_change_points)

    if 0 not in all_change_points:
        all_change_points.append(0)
    if len(volume) not in all_change_points:
        all_change_points.append(len(volume))

    all_change_points = sorted(all_change_points)
    merged_change_points = merge_trend_points(all_change_points, volume, avg_threshold, slope_threshold)

    final_change_points = [merged_change_points[0]]

    for i in range(len(merged_change_points) - 1):
        start_idx = merged_change_points[i]
        end_idx = merged_change_points[i + 1]
        segment = volume[start_idx:end_idx]
        if len(segment) > 0:
            trend = "UP" if segment[-1] > segment[0] else "DOWN"

            if len(final_change_points) >= 2 and i > 0:
                prev_start_idx = final_change_points[-2]
                prev_end_idx = final_change_points[-1]
                prev_segment = volume[prev_start_idx:prev_end_idx]

                if len(prev_segment) > 0:
                    prev_trend = "UP" if prev_segment[-1] > prev_segment[0] else "DOWN"

                    if prev_trend == trend:
                        final_change_points.pop()

            final_change_points.append(end_idx)

    liq_data_dir = f'./MyData/{dataset}/liq_data/{model}/{data_type}'
    os.makedirs(liq_data_dir, exist_ok=True)

    liq_labels = []
    liq_files = []

    for i in range(len(final_change_points) - 1):
        start_idx = final_change_points[i]
        end_idx = final_change_points[i + 1]
        segment = df.iloc[start_idx:end_idx].copy()
        segment = segment.reset_index(drop=True)

        volume_segment = volume[start_idx:end_idx]
        liq_label = 1 if volume_segment[-1] > volume_segment[0] else -1

        filename = f'df_{i}.feather'
        filepath = os.path.join(liq_data_dir, filename)
        segment.to_feather(filepath)

        liq_labels.append(liq_label)
        liq_files.append(filename)

    labels_dict = {-1: [], 1: []}
    for i, label in enumerate(liq_labels):
        labels_dict[label].append(i)

    with open(os.path.join(liq_data_dir, 'liq_labels.pkl'), 'wb') as f:
        pickle.dump(labels_dict, f)

    print(f"\nLiquidity {data_type}******** Total segments: {len(liq_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='BTCUSDT')
    parser.add_argument("--method", type=str, default='linear')

    args, remaining = parser.parse_known_args()

    print(args.method)

    if args.method == 'linear':

        parser.add_argument("--trend_method", type=str, default='linear')
        parser.add_argument("--trend_penalty", type=float, default=0.00000000000000001)
        parser.add_argument("--trend_avg_threshold", type=float, default=0.0065)
        parser.add_argument("--trend_slope_threshold", type=float, default=0.000005)
        parser.add_argument("--vol_method", type=str, default='linear')
        parser.add_argument("--vol_penalty", type=float, default=0.000000000000000001)
        parser.add_argument("--vol_threshold", type=float, default=0.004)
        parser.add_argument("--liq_method", type=str, default='linear')
        parser.add_argument("--liq_penalty", type=float, default=0.0000000000000001)
        parser.add_argument("--liq_avg_threshold", type=float, default=0.06)
        parser.add_argument("--liq_slope_threshold", type=float, default=0.0005)

    elif args.method == 'l2':
        parser.add_argument("--trend_method", type=str, default='l2')
        parser.add_argument("--trend_penalty", type=float, default=0.00000001)
        parser.add_argument("--trend_avg_threshold", type=float, default=0.0075)
        parser.add_argument("--trend_slope_threshold", type=float, default=0.000008)
        parser.add_argument("--vol_method", type=str, default='l2')
        parser.add_argument("--vol_penalty", type=float, default=0.00001)
        parser.add_argument("--vol_threshold", type=float, default=0.004)
        parser.add_argument("--liq_method", type=str, default='l2')
        parser.add_argument("--liq_penalty", type=float, default=0.000000001)
        parser.add_argument("--liq_avg_threshold", type=float, default=0.06)
        parser.add_argument("--liq_slope_threshold", type=float, default=0.0006)


    elif args.method == 'l1':
        parser.add_argument("--trend_method", type=str, default='l1')
        parser.add_argument("--trend_penalty", type=float, default=0.001)
        parser.add_argument("--trend_avg_threshold", type=float, default=0.0075)
        parser.add_argument("--trend_slope_threshold", type=float, default=0.000008)
        parser.add_argument("--vol_method", type=str, default='l1')
        parser.add_argument("--vol_penalty", type=float, default=0.001)
        parser.add_argument("--vol_threshold", type=float, default=0.004)
        parser.add_argument("--liq_method", type=str, default='l1')
        parser.add_argument("--liq_penalty", type=float, default=0.002)
        parser.add_argument("--liq_avg_threshold", type=float, default=0.065)
        parser.add_argument("--liq_slope_threshold", type=float, default=0.0005)

    elif args.method == 'rbf':
        parser.add_argument("--trend_method", type=str, default='rbf')
        parser.add_argument("--trend_penalty", type=float, default=0.005)
        parser.add_argument("--trend_avg_threshold", type=float, default=0.007)
        parser.add_argument("--trend_slope_threshold", type=float, default=0.0000075)
        parser.add_argument("--vol_method", type=str, default='rbf')
        parser.add_argument("--vol_penalty", type=float, default=0.0015)
        parser.add_argument("--vol_threshold", type=float, default=0.0045)
        parser.add_argument("--liq_method", type=str, default='rbf')
        parser.add_argument("--liq_penalty", type=float, default=0.005)
        parser.add_argument("--liq_avg_threshold", type=float, default=0.065)
        parser.add_argument("--liq_slope_threshold", type=float, default=0.0005)

    args = parser.parse_args()

    df_train = pd.read_feather(f'./MyData/{args.dataset}/df_train.feather')
    df_val = pd.read_feather(f'./MyData/{args.dataset}/df_val.feather')
    df_test = pd.read_feather(f'./MyData/{args.dataset}/df_test.feather')
    df_whole = pd.read_feather(f'./MyData/{args.dataset}/df_whole.feather')

    os.makedirs(f'./MyData/{args.dataset}/trend_data', exist_ok=True)
    os.makedirs(f'./MyData/{args.dataset}/vol_data', exist_ok=True)
    os.makedirs(f'./MyData/{args.dataset}/liq_data', exist_ok=True)
    os.makedirs(f'./MyData/{args.dataset}/whole', exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        print("Starting parallel labeling...")

        futures = [
            executor.submit(label_trend_pelt, df_train, args, "train"),
            executor.submit(label_trend_pelt, df_val, args, "val"),
            executor.submit(label_trend_pelt, df_test, args, "test"),
            executor.submit(label_volatility_pelt, df_train, args, "train"),
            executor.submit(label_volatility_pelt, df_val, args, "val"),
            executor.submit(label_volatility_pelt, df_test, args, "test"),
            executor.submit(label_liquidity_pelt, df_train, args, "train"),
            executor.submit(label_liquidity_pelt, df_val, args, "val"),
            executor.submit(label_liquidity_pelt, df_test, args, "test")
        ]

        concurrent.futures.wait(futures)

    df_train.reset_index(drop=True).to_feather(f'./MyData/{args.dataset}/whole/train.feather')
    df_val.reset_index(drop=True).to_feather(f'./MyData/{args.dataset}/whole/val.feather')
    df_test.reset_index(drop=True).to_feather(f'./MyData/{args.dataset}/whole/test.feather')

    print("finished")
