import tensorflow as tf
from tensorflow.keras import backend as K


def weighted_huber_loss(delta=1.5, weight=1.5):
    def huber_loss(y_true, y_pred):
        error = (y_pred - y_true)
        abs_error = tf.math.abs(error)
        wrong_side_mask = tf.cast(tf.sign(y_true) != tf.sign(y_pred), tf.float32)
        quadratic = tf.math.minimum(abs_error, delta)
        linear = abs_error - quadratic
        h_loss = .5 * tf.math.exp(quadratic) ** 2 + delta * linear
        weight_h_loss = (1 + wrong_side_mask * (weight - 1)) * h_loss

        return tf.reduce_mean(weight_h_loss)

    return huber_loss


def weighted_categorical_crossentropy(class_weights):
    def wce_loss(y_true, y_pred):
        weights_tensor = tf.constant(class_weights, dtype=tf.float32)
        weights_tensor = tf.reshape(weights_tensor, (1, -1))  # Shape: (1, num_classes)

        weights = tf.reduce_sum(weights_tensor * y_true, axis=-1)  # Shape: (batch_size,)
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        weighted_loss = tf.reduce_mean(weights * cross_entropy)

        return weighted_loss

    return wce_loss


def bal_accuracy(y_true, y_pred):
    y_pred_binary = tf.cast(y_pred >= 0.5, tf.float32)

    true_positives = K.sum(K.cast(y_true * y_pred_binary, tf.float32))
    true_negatives = K.sum(K.cast((1 - y_true) * (1 - y_pred_binary), tf.float32))
    false_positives = K.sum(K.cast((1 - y_true) * y_pred_binary, tf.float32))
    false_negatives = K.sum(K.cast(y_true * (1 - y_pred_binary), tf.float32))

    recall_pos = tf.clip_by_value(true_positives / (true_positives + false_negatives + K.epsilon()),
                                  K.epsilon(), 1 - K.epsilon())
    recall_neg = tf.clip_by_value(true_negatives / (true_negatives + false_positives + K.epsilon()),
                                  K.epsilon(), 1 - K.epsilon())

    balanced_acc = (recall_pos + recall_neg) / 2.0
    return (1 - balanced_acc) + K.epsilon()


def focal_loss(gamma=1.5, alpha=0.3):
    def focal_loss_fixed(y_true, y_pred):
        gamma_value = gamma
        alpha_value = alpha

        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

        cross_entropy_pos = -y_true * tf.math.pow(1 - y_pred, gamma_value) * tf.math.log(y_pred)
        cross_entropy_neg = -(1 - y_true) * tf.math.pow(y_pred, gamma_value) * tf.math.log(1 - y_pred)

        loss = tf.sqrt(alpha_value * cross_entropy_pos + (1 - alpha_value) * cross_entropy_neg)

        return tf.reduce_mean(loss)
    return focal_loss_fixed


def beta_f1(beta=1.0, opt_threshold=0.5):
    b2 = beta**2

    def f1_score(y_true, y_pred):
        y_pred = tf.cast(y_pred >= opt_threshold, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum(y_pred) - tp
        fn = tf.reduce_sum(y_true) - tp

        precision = tp / (tp + fp + tf.keras.backend.epsilon())
        recall = tp / (tp + fn + tf.keras.backend.epsilon())
        f1 = (1 + b2) * (precision * recall) / (b2 * precision + recall + tf.keras.backend.epsilon())

        return f1

    return f1_score


def negative_predictive_value(opt_threshold=0.5):
    def npv(y_true, y_pred):
        y_pred = tf.cast(y_pred >= opt_threshold, tf.float32)
        true_negatives = tf.reduce_sum((1 - y_true) * (1 - y_pred))
        false_negatives = tf.reduce_sum(y_true * (1 - y_pred))

        np_value = true_negatives / (true_negatives + false_negatives + tf.keras.backend.epsilon())

        return np_value

    return npv


def positive_predictive_value(opt_threshold=0.5):
    def ppv(y_true, y_pred):
        y_pred = tf.cast(y_pred >= opt_threshold, tf.float32)
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum(y_pred) - tp

        np_value = tp / (tp + fp + tf.keras.backend.epsilon())

        return np_value

    return ppv


def weighted_auc(class_weights):
    neg_weight = class_weights[0]
    pos_weight = class_weights[1]

    def auc_loss(y_true, y_pred):
        """
        AUC loss function: minimizes the pairwise ranking loss to approximate AUC.
        y_true: Ground truth labels, expected to be 0 or 1.
        y_pred: Predicted probabilities (output of the model).
        """
        pos_mask = tf.equal(tf.argmax(y_true, axis=-1), 1)
        neg_mask = tf.equal(tf.argmax(y_true, axis=-1), 0)

        pos_pred = tf.boolean_mask(y_pred[:, 1], pos_mask)
        neg_pred = tf.boolean_mask(y_pred[:, 1], neg_mask)

        def safe_mean(tensor):
            return tf.reduce_mean(tensor) if tf.size(tensor) > 0 else tf.constant(0.0, dtype=tf.float32)

        pairwise_diff = tf.expand_dims(pos_pred, axis=1) - tf.expand_dims(neg_pred, axis=0)

        surrogate_loss = tf.nn.sigmoid(-pairwise_diff)

        pair_weights = pos_weight * neg_weight
        weighted_loss = surrogate_loss * pair_weights

        loss = safe_mean(weighted_loss)

        return loss

    return auc_loss


"""-------------------------------------------Combined Functions-----------------------------------------------------"""


def comb_focal_wce_f1(beta=1.0, opt_threshold=0.5, class_weights=(0.9, 1.5)):
    loss_wce_fn = weighted_categorical_crossentropy(class_weights)
    f1_score_fn = beta_f1(beta, opt_threshold)
    auc_fn = weighted_auc(class_weights)
    npv_fn = negative_predictive_value(opt_threshold)
    ppv_fn = positive_predictive_value(opt_threshold)

    def combined_wl_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Calculate individual losses
        loss_fl = focal_loss()(y_true, y_pred)
        wce_loss = loss_wce_fn(y_true, y_pred)
        f1_loss = tf.math.log1p(1 - f1_score_fn(y_true, y_pred))
        auc_loss = auc_fn(y_true, y_pred)
        npv_loss = tf.math.log1p(1 - npv_fn(y_true, y_pred))
        ppv_loss = tf.math.log1p(1 - ppv_fn(y_true, y_pred))

        # Weighted sum of the losses
        weight_fl = 1.0
        weight_f1 = 0.5
        weight_wce = 2.0
        weight_auc = 0.5
        weight_npv = 0.5
        weight_ppv = 0.5

        combined_loss_value = (weight_fl * loss_fl + weight_wce * wce_loss +
                               weight_f1 * f1_loss + weight_auc * auc_loss +
                               weight_npv * npv_loss + weight_ppv * ppv_loss)

        return combined_loss_value

    return combined_wl_loss
