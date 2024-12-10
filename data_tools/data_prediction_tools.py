import numpy as np
import pandas as pd
import data_tools.math_tools as mt
from nn_tools.model_tools import CustomDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats

pd.set_option('display.max_columns', None)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nn_tools.process_handler import ProcessHandler


class ModelOutputData:
    def __init__(self, process_handler: "ProcessHandler"):
        self.ph = process_handler
        self.ph.model_output_data = self
        self.model_metrics = []
        self.wl_loss = None
        self.wl_nn_binary = None
        self.wl_algo_binary = None
        self.optimal_threshold = None
        self.optimized_temperature = None

