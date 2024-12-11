import pandas as pd
import numpy as np
from numba import jit
from arch import arch_model


def create_atr(df, sec, n=8):
    high_low = df[f'{sec}_High'] - df[f'{sec}_Low']
    high_prev_close = np.abs(df[f'{sec}_High'] - df[f'{sec}_Close'].shift(1))
    low_prev_close = np.abs(df[f'{sec}_Low'] - df[f'{sec}_Close'].shift(1))
    true_range = np.maximum(high_low, high_prev_close)
    true_range = np.maximum(true_range, low_prev_close)

    # Calculate Average True Range (ATR)
    atr = np.zeros_like(df[f'{sec}_Close'])
    atr[n - 1] = np.mean(true_range[:n])  # Initial ATR calculation

    for i in range(n, len(df[f'{sec}_Close'])):
        atr[i] = ((atr[i - 1] * (n - 1)) + true_range[i]) / n

    return atr


def create_rsi(series, period=14):
    # Calculate price differences
    delta = series.diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Compute exponential moving averages of gains and losses
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def create_smooth_rsi(series, period=14, smooth=9):
    rsi = create_rsi(series, period)
    rsi_temp = pd.Series(rsi)
    rsi_smooth = rsi_temp.rolling(window=smooth).mean()

    return rsi, rsi_smooth


def add_high_low_diff(df, sec):
    df[f'{sec}_HL_diff'] = (
            df[f'{sec}_High'] - df[f'{sec}_Low']) / ((df[f'{sec}_High'] + df[f'{sec}_Low'])/2)*1000
    df[f'{sec}_OC_diff'] = (
            df[f'{sec}_Open'] - df[f'{sec}_Close']) / ((df[f'{sec}_Open'] + df[f'{sec}_Close'])/2)*1000

    df[f'{sec}_HL_Ratio'] = df[f'{sec}_HL_diff'] / df[f'{sec}_HL_diff'].shift(1)
    df[f'{sec}_OC_Ratio'] = df[f'{sec}_OC_diff'] / df[f'{sec}_OC_diff'].shift(1)

    return df


def calculate_ema_numba(df, price_colname, window_size, smoothing_factor=2):
    result = calculate_ema_inner(
        price_array=df[price_colname].to_numpy(),
        window_size=window_size,
        smoothing_factor=smoothing_factor
    )

    return result


@jit(nopython=True)
def calculate_ema_inner(price_array, window_size, smoothing_factor):
    result = np.empty(len(price_array), dtype="float64")
    sma_list = list()
    for i in range(len(result)):

        if i < window_size - 1:
            result[i] = np.nan
            sma_list.append(price_array[i])
        elif i == window_size - 1:
            sma_list.append(price_array[i])
            result[i] = sum(sma_list) / len(sma_list)
        else:
            result[i] = ((price_array[i] * (smoothing_factor / (window_size + 1))) +
                         (result[i - 1] * (1 - (smoothing_factor / (window_size + 1)))))

    return result


def standardize_ema(arr, lag=12):
    arr = np.array(arr)
    standardized_arr = np.ones_like(arr, dtype=np.float32)

    standardized_arr[lag:] = arr[lag:] / arr[:-lag]

    return standardized_arr


def encode_time_features(df: pd.DataFrame,
                         time_frame: str):
    # Cyclic encoding for Month, Day, Hour
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)

    if time_frame != 'daily':
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Minute_sin'] = np.cos(2 * np.pi * df['Minute'] / 60)
        df['Minute_cos'] = np.cos(2 * np.pi * df['Minute'] / 24)

    for col in ['Month', 'Day', 'Hour', 'Minute']:
        if col in df.columns:
            df = df.drop(columns=col, errors='ignore')

    return df


def subset_to_first_nonzero(arr):
    first_nonz_ind = np.argmax(arr != 0)
    trimmed_arr = arr[first_nonz_ind:]

    return trimmed_arr


def garch_modeling(df, sec):
    print(f'Modeling EGARCH')
    for met in ['Close', 'Vol']:
        print(f'...{sec}')
        temp_df = df[f'{sec}_{met}']
        temp_df = rescale_data_to_range(temp_df, 500)
        # garch_m = arch_model(temp_df, vol='GARCH', p=1, q=1)
        # garch_fit = garch_m.fit(disp='off')
        # df[f'{sec}_{met}_garch_cv'] = garch_fit.conditional_volatility
        # df[f'{sec}_{met}_garch_std'] = garch_fit.std_resid

        garch_m = arch_model(temp_df, vol='EGARCH', p=1, o=1, q=1)
        garch_fit = garch_m.fit(disp='off')
        df[f'{sec}_{met}_egarch_cv'] = garch_fit.conditional_volatility
        df[f'{sec}_{met}_egarch_std'] = garch_fit.std_resid

    return df


def rescale_data_to_range(df, max_range=1000):
    if df.abs().max() > max_range:
        scale_factor = max_range / df.abs().max()
        temp_df_scaled = df * scale_factor
    else:
        temp_df_scaled = df

    return temp_df_scaled

