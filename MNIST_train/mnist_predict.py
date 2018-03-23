import tensorflow as tf
from predImage import imageprepare

def predictint(imvalue):
    """
    This function returns the predicted integer.
    The imput is the pixel values from the imageprepare() function.
    """
   
    tf.reset_default_graph()
    
    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "D:\\testTensorFlow\\MNIST_data\\model1.ckpt")
        #print ("Model restored.")
   
        prediction=tf.argmax(y,1)
        data = prediction.eval(feed_dict={x: [imvalue]}, session=sess)
        sess.close()
        return data
    
if __name__=='__main__':
    d = imageprepare("20736.png")
    print(d)
    data = predictint(d)
    print(data)