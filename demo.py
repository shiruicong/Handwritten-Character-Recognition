import tensorflow as tf
import cv2
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(train_dir="data", one_hot=True)
batch_size = 1000
test_x, test_y = mnist.test.next_batch(batch_size)
# input_image = "./data/111.jpg"
# gray = cv2.cvtColor(cv2.imread(test_x), cv2.COLOR_BGR2GRAY)
# size = gray.shape
# temp = cv2.resize(gray, (28, 28))
# cv2.imshow('image', temp)
# temp = np.reshape(temp, (-1, 784))
with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./model/mnist.ckpt-200.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./model'))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_pre = graph.get_tensor_by_name("y_label:0")
    label = sess.run(y_pre, feed_dict={x: test_x})
    num = tf.equal(label, tf.arg_max(test_y, 1))
    correct_prediction = tf.reduce_sum(tf.cast(num, tf.float32))/10
    accracy = sess.run(correct_prediction, feed_dict={x: test_x})
    print(accracy)


