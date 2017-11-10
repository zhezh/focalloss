import tensorflow as tf
import numpy as np
from focalloss import focal_loss


fl_gamma = 2.0
fl_alpha = 4.0


if __name__ == '__main__':
    model_out = tf.constant([[0.1, 0.7, 0.1, 0.05, 0.05],
                             [0.01, 0.1, 0.79, 0.05, 0.05]], dtype=tf.float32)
    labels = tf.constant([1, 1], dtype=tf.int64)
    # labels = tf.constant([1., 1.], dtype=tf.float32)  # also works

    func_fl = focal_loss(labels, model_out, fl_gamma, fl_alpha)
    loss = tf.reduce_mean(func_fl)

    with tf.Session() as sess:
        _fl, _loss = sess.run([func_fl, loss])
        print('focal loss: ', _fl)
        print('reduced loss: ', _loss)

    grad = tf.gradients(func_fl, model_out)
    with tf.Session() as sess:
        _gra = sess.run([grad])
        print('gradient: ',_gra)