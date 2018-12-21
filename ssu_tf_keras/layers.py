import tensorflow as tf
import numpy as np

class Reverse(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.reverse(inputs, self.axis)

class ExpandDims(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        return tf.expand_dims(inputs, axis=self.axis)

class ReflectionPadding2D(tf.keras.layers.ZeroPadding2D):
   def call(self, inputs, mask=None):
       #print(self.padding)
       pattern = [[0, 0],
                  self.padding[0],
                  self.padding[1],
                  [0, 0]]
       return tf.pad(inputs, pattern, mode='REFLECT')

if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    x = tf.placeholder(shape=[None, 5, 5, 3], dtype=tf.int32, name='x')
    inputs = tf.keras.Input(shape=[5,5,3])

    rp = ReflectionPadding2D([2,2])(inputs)
    rp_model = tf.keras.Model(inputs, rp)
    rp_out = rp_model(x)

    ed = ExpandDims(axis=-1)(inputs)
    ed_model = tf.keras.Model(inputs, ed)
    ed_out = ed_model(x)

    rv = Reverse([1])(inputs)
    rv_model = tf.keras.Model(inputs, rv)
    rv_out = rv_model(x)


    arr = [
            [
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]],
            [[10, 10, 10], [11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14]],
            [[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]],
            [[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]],
            ],
            [
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]],
            [[10, 10, 10], [11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14]],
            [[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]],
            [[5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]],
            ]
        ]

    print(np.array(arr).shape)
    print('original input: {}'.format(arr))
    print('------------------')
    print('test Reflection Padding that shape: {}'.format(sess.run(tf.shape(rp_out),feed_dict = {x: arr})))
    print('test Expand Dims that shape: {}'.format(sess.run(tf.shape(ed_out), feed_dict={x: arr})))
    print('test Reverse: {}'.format(sess.run(ed_out, feed_dict={x: arr})))








