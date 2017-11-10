import tensorflow as tf


def focal_loss(labels, logits, gamma=2.0, alpha=4.0):
    """
    focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: logits is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)

    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection, 130(4), 485–491.
    https://doi.org/10.1016/j.ajodo.2005.02.022

    :param labels: ground truth labels, shape of [batch_size]
    :param logits: model's output, shape of [batch_size, num_cls]
    :param gamma:
    :param alpha:
    :return: shape of [batch_size]
    """
    epsilon = 1.e-9
    labels = tf.to_int64(labels)
    labels = tf.convert_to_tensor(labels, tf.int64)
    logits = tf.convert_to_tensor(logits, tf.float32)
    num_cls = logits.shape[1]

    model_out = tf.add(logits, epsilon)
    onehot_labels = tf.one_hot(labels, num_cls)
    ce = tf.multiply(onehot_labels, -tf.log(model_out))
    weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    # reduced_fl = tf.reduce_sum(fl, axis=1)  # same as reduce_max
    return reduced_fl
