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
        self.model_metrics = []

    def predict_data(self):
        test_gen = glt.BufferedBatchGenerator(self.ph, 100, train=False)
        test_gen = test_gen.load_full_dataset()
        predictions = self.ph.lstm_model.model.predict(test_gen)
        preds_labeled = self.ph.lstm_data.y_wl_onehot_scaler.inverse_transform(predictions)
        check = self.ph.trade_data.analysis_df.copy(deep=True)
        check = check.iloc[-len(preds_labeled):]
        check['pred_labeled'] = preds_labeled.flatten()
        # for i in range(4):
        #     check[i] = predictions[:, i]
        check.to_excel('check.xlsx')
        labels = np.unique(self.ph.trade_data.y_train_df['Label'])
        for i in range(predictions.shape[1]):
            check[labels[i]] = predictions[:, i]
        check.to_excel('check.xlsx')

