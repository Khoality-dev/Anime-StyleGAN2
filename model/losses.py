import tensorflow as tf

def gradient_penalty(Discriminator, reals, fakes):
    batch_size = len(reals)
    H, W = len(reals[0]), len(reals[0][0])
    with tf.GradientTape() as tape:
        epsilon = tf.random.uniform(shape = (batch_size,1,1,1))
        epsilon = tf.tile(epsilon, [1,H,W,3])
        tf_reals = tf.Variable(reals, dtype = float)
        tf_fakes = tf.Variable(fakes)
        inter_images = tf.multiply(tf_reals,epsilon) + tf.multiply(tf_fakes,1-epsilon)
        mixed_score = Discriminator(inter_images, training = True)
        grad = tape.gradient(mixed_score, inter_images)
        grad = tf.norm(grad, 2)
        grad = tf.reduce_mean((grad - 1) ** 2)
    return grad

def WGAN_Disciminator_loss(Discriminator, reals, fakes):
    lamda = 10
    ans = tf.reduce_mean(Discriminator(reals, training = True)) - tf.reduce_mean(Discriminator(fakes, training = True))
    ans = ans - lamda * gradient_penalty(Discriminator, reals, fakes)
    return -ans 

#y_true is dummy values, not used in calculation
def WGAN_Generator_Loss(y_pred, y_true = None):
    ans = tf.reduce_mean(y_pred)
    return -ans