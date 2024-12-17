import numpy as np
import pandas as pd
import data_tools.math_tools as mt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats
import nn_tools.general_lstm_tools as glt

pd.set_option('display.max_columns', None)


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nn_tools.process_handler import ProcessHandler


class ModelOutputData:
    def __init__(self, process_handler: "ProcessHandler"):
        self.ph = process_handler
        self.ph.model_output_data = self
        self.x_test_data = pd.DataFrame
        self.model_metrics = initial_model_metrics()

    def predict_data(self):
        test_gen = glt.BufferedBatchGenerator(self.ph, train=False)
        test_gen = test_gen.load_full_dataset()

        test_loss = self.ph.lstm_model.model.evaluate(test_gen)

        self.model_metrics['test_loss'] = test_loss

        for i in range(self.ph.lstm_model.lstm_dict['prediction_runs']):
            batch_preds = []
            for data_x, _ in test_gen:
                predictions = self.ph.lstm_model.model(data_x, training=True)
                batch_preds.append(predictions)
            batch_preds = np.concatenate(batch_preds, axis=0)
            self.model_metrics['probability_preds'].append(batch_preds)
            # predictions = self.ph.lstm_model.model.predict(test_gen)
            # self.model_metrics['probability_preds'].append(predictions)

        self.agg_prediction_data()

        preds_labeled = self.ph.lstm_data.y_wl_onehot_scaler.inverse_transform(self.model_metrics['probability_preds'])
        self.model_metrics['labeled_preds'] = preds_labeled.flatten()
        # print('labeled_preds')
        # print(self.model_metrics['labeled_preds'])
        # print(len(self.model_metrics['labeled_preds']))

    def agg_prediction_data(self):
        pred_arr = np.array(self.model_metrics['probability_preds'])
        # pred_arr = np.mean(pred_arr, axis=0)
        num_class = pred_arr.shape[2]
        pred_arr = np.argmax(pred_arr, axis=2)
        pred_arr = np.eye(num_class)[pred_arr]
        # pred_arr = pred_arr / pred_arr.sum(axis=2, keepdims=True)
        pred_arr = np.mean(pred_arr, axis=0)
        self.model_metrics['probability_preds'] = pred_arr

    def prediction_analysis(self):
        labeled_preds = self.model_metrics['labeled_preds']
        anal_df = self.ph.trade_data.analysis_df.copy(deep=True)
        anal_df = build_analysis_df(anal_df, labeled_preds)
        one_dir_algo_stats = lstm_trade_stats(anal_df, pred_data=False, two_dir=False)
        one_dir_pred_stats = lstm_trade_stats(anal_df, pred_data=True, two_dir=False)
        two_dir_pred_stats = lstm_trade_stats(anal_df, pred_data=True, two_dir=True)

        predicted_trade_data = [anal_df], [one_dir_algo_stats, one_dir_pred_stats, two_dir_pred_stats]

        return predicted_trade_data


def build_analysis_df(df, labeled_preds):
    df = df.iloc[-len(labeled_preds):]
    df['Lstm_label'] = labeled_preds

    df['PnL_algo_tot'] = df['PnL'].cumsum()
    df['Maxdraw_algo'] = mt.calculate_max_drawdown(df['PnL_algo_tot'])

    df['PnL_one_dir'] = df['PnL'].where(df['Lstm_label'] == 'lg_win', 0)
    df['PnL_one_dir_tot'] = df['PnL_one_dir'].cumsum()
    df['Maxdraw_one_dir'] = mt.calculate_max_drawdown(df['PnL_one_dir_tot'])

    df['PnL_two_dir'] = 0
    df['PnL_two_dir'] = df.apply(
        lambda row: row['PnL'] if row['Lstm_label'] == 'lg_win' else
        -row['PnL'] if row['Lstm_label'] == 'lg_loss' else 0,
        axis=1)
    df['PnL_two_dir_tot'] = df['PnL_two_dir'].cumsum()
    df['Maxdraw_two_dir'] = mt.calculate_max_drawdown(df['PnL_two_dir'])

    return df


def lstm_trade_stats(df, pred_data=True, two_dir=True):
    algo_lab_arr = np.array(df['Algo_label'])
    lstm_lab_arr = np.array(df['Lstm_label'])
    unique_labels = np.unique(np.concatenate((algo_lab_arr, lstm_lab_arr)))

    if pred_data:
        match_label = {label: np.sum((algo_lab_arr == label) & (lstm_lab_arr == label)) for label in unique_labels}
        model_dat = 'Lstm_label'
    else:
        match_label = {label: np.sum((algo_lab_arr == label)) for label in unique_labels}
        model_dat = 'Algo_label'

    results = {}
    for key in match_label.keys():
        num_algo_labels = np.sum(df['Algo_label'] == key)
        num_correct = match_label[key]

        if model_dat == 'Lstm_label':
            percent_correct = num_correct / num_algo_labels if num_algo_labels > 0 else 0
        else:
            percent_correct = (
                    np.sum((df['Algo_label'] == key) & (df['PnL'] > 0)) / len(df.index)) if num_algo_labels > 0 else 0

        total_pnl = np.sum(df[df[model_dat] == key]['PnL'])

        num_trades = len(df[df[model_dat] == key])
        avg_trade = total_pnl / num_trades if num_trades > 0 else 0

        # Average win and loss
        pnl_wins = df[(df[model_dat] == key) & (df['PnL'] > 0)]['PnL']
        pnl_losses = df[(df[model_dat] == key) & (df['PnL'] <= 0)]['PnL']
        avg_win = np.mean(pnl_wins) if len(pnl_wins) > 0 else 0
        avg_loss = np.mean(pnl_losses) if len(pnl_losses) > 0 else 0

        if key in ['lg_win', 'sm_win']:
            exp_value = (percent_correct * avg_win) + ((1 - percent_correct) * avg_loss)
        else:
            exp_value = ((1 - percent_correct) * avg_win) + (percent_correct * avg_loss)

        # max_draw = np.min(df['Maxdraw_algo'])
        # pnl = df['PnL']
        # if pred_data:
        #     if two_dir:
        #         max_draw = np.min(df['Maxdraw_one_dir'])
        #         pnl = df['PnL_two_dir']
        #     else:
        #         max_draw = np.min(df['Maxdraw_two_dir'])
        #         pnl = df['PnL_one_dir']
        #
        # max_pnl = np.max(pnl)
        # sortino = mt.sortino_ratio(pnl)

        results[key] = {
            '%_correct': percent_correct,
            'total_pnl': total_pnl,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expected_value': exp_value,
            # 'max_pnl': max_pnl,
            # 'max_draw': max_draw,
            # 'sortino': sortino
        }

    if two_dir and pred_data:
        if not 'skip' in results.keys():
            for key in ['lg_loss', 'sm_loss']:
                for metric in ['total_pnl', 'avg_trade', 'expected_value']:
                    if results[key][metric] != 0:
                        results[key][metric] = -results[key][metric]
                avg_win = results[key]['avg_win']
                results[key]['avg_win'] = -results[key]['avg_loss']
                results[key]['avg_loss'] = -avg_win

    results = pd.DataFrame(results)

    return results


def initial_model_metrics():
    metrics_dict = {
        'test_loss': 0.0,
        'probability_preds': [],
        'labeled_preds': []
    }

    return metrics_dict


