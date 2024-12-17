import pandas as pd
import numpy as np


file_fold = r'C:\Users\jmdub\Documents\Trading\Futures\Strategy Info\Double_Candles\ATR\NQ\15min\15min_test_20years'
file_loc = f'{file_fold}\\NQ_15min_Double_Candle_289_trades.feather'


def set_pnl(df):
    df['PnL'] = np.where(df['side'] == 'Bear',
                         df['entryPrice'] - df['exitPrice'],
                         df['exitPrice'] - df['entryPrice'])

    return df


def set_labels(df, side, low_percentile, high_percentile):
    df['PnL'] = np.where(df['side'] == 'Bear',
                         df['entryPrice'] - df['exitPrice'],
                         df['exitPrice'] - df['entryPrice'])
    df['Label'] = np.empty(len(df), dtype=object)
    df = df[df['side'] == side]
    temp_train = df[df['DateTime'] <= pd.to_datetime('2021-04-01')]
    pnl_arr = temp_train['PnL'].values
    percentile_low = np.percentile(pnl_arr, low_percentile)
    percentile_high = np.percentile(pnl_arr, high_percentile)

    conds = [(df['PnL'] <= percentile_low),
             (df['PnL'] < percentile_high) &
             (df['PnL'] > percentile_low),
             (df['PnL'] >= percentile_high)]
    labels = ['lg_loss', 'skip', 'lg_win']
    default = 'skip'

    df.loc[df.index, 'Label'] = np.select(conds, labels, default=default)

    return df


def compute_penalty_matrix(df, classes):
    penalty_matrix = np.zeros((len(classes), len(classes)))
    avg_pnl_per_class = df.groupby("Label")["PnL"].mean()
    avg_pnl_per_class = avg_pnl_per_class.reindex(classes, fill_value=0)

    for i, cls_true in enumerate(classes):
        for j, cls_pred in enumerate(classes):
            if i != j:
                penalty_matrix[i, j] = abs(avg_pnl_per_class[cls_true] - avg_pnl_per_class[cls_pred])

    lower_triangle_sum = np.sum(np.tril(penalty_matrix, k=-1))
    if lower_triangle_sum == 0:
        print("The sum of the lower triangle in the penalty matrix is zero. Cannot normalize.")
        return penalty_matrix

    penalty_matrix /= lower_triangle_sum
    penalty_matrix += 1

    p_matrix = pd.DataFrame(penalty_matrix, index=classes, columns=classes)
    return p_matrix


trade_df = pd.read_feather(f'{file_loc}')
trade_df = set_pnl(trade_df)
trade_df = set_labels(trade_df, 'Bull', 20, 25)
classes = ['lg_loss', 'skip', 'lg_win']
p_mat = compute_penalty_matrix(trade_df, classes)
print(p_mat)

