import tensorflow as tf
import keras.backend as K

class AdaIN(tf.keras.layers.Layer):
    def __init__(self):
        super(AdaIN, self).__init__()
    
    def call(self, inputs):
        x = inputs[0]

        beta_1 = inputs[1]
        beta_2 = inputs[2]

        mean = tf.reduce_mean(x, axis = (1,2), keepdims=True)
        stddev = tf.math.reduce_std(x, axis = (1,2), keepdims=True)
        normalized_x = (x - mean) / (stddev + 1e-8)

        return normalized_x * beta_2 + beta_1


class Minibatch_stddev(tf.keras.layers.Layer):
    def __init__(self):
        super(Minibatch_stddev, self).__init__()
    
    def call(self, inputs):
        shape = tf.shape(inputs)
        std = tf.math.reduce_std(inputs, axis=0, keepdims=True)
        mean_std = tf.reduce_mean(std,keepdims=True)
        output = tf.tile(mean_std, [shape[0], shape[1], shape[2], 1])
        return tf.concat([inputs, output], axis = -1)



class Bias_Layer(tf.keras.layers.Layer):
    def __init__(self):
        super(Bias_Layer, self).__init__()
    
    def build(self, input_shape):
        self.bias = self.add_weight('bias', shape=input_shape[1:], initializer='ones', trainable=True)
    
    def call(self, x):
        return x + self.bias
