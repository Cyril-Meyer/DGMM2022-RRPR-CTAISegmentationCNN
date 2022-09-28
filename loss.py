import tensorflow as tf


def dice_coef_tf(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(tf.multiply(y_true_f, y_pred_f))
    smooth = 0.0001
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_tf_2_classes(y_true, y_pred):
    dice = 0
    for index in range(2):
        dice += dice_coef_tf(y_true[..., index], y_pred[..., index])
    return dice / 2
