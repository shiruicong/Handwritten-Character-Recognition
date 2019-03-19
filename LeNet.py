import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# mist = tf.contrib.learn.datasets.read_data_sets(train_dir="data", one_hot=True)
mnist = input_data.read_data_sets(train_dir="data", one_hot=True)
# tf.nn.conv2d的参数解释
# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
# input：输入图像，四维的Tensor，[batch, in_height, in_width, in_channels]
# filter：卷积核，tensor类型，具有[filter_height, filter_width, in_channels, out_channels]
# strides：步长 strides[0]=1第一维度，步长为1
# padding：填充，string类型“same:0填充”或“valid：不填充"
# cudnn:是否使用cudnn加速，默认为true
#  最后返回一个张量，就是特征图feature map

def conv_op(input,name,kernel_h,kernel_w,out_channels,):

    # 卷积层
    in_channels = input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(name=scope+"w", shape=[kernel_h, kernel_w, in_channels, out_channels],
                             dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input=input, filter=kernel, strides=(1, 1, 1, 1), padding="VALID")
        # biases = tf.ones_initializer(shape=[out_channels])
        b_init = tf.constant(0.0, shape=[out_channels], dtype=tf.float32)
        biases = tf.Variable(b_init, trainable=True, name="biases")
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        output = tf.nn.max_pool(value=activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        return output

def LENet(input_image,lable):

    input = tf.reshape(input_image, [-1, 28, 28, 1])
    conv1 = conv_op(input, "conv1", 5, 5, 16)  # 24*24*32
    conv2 = conv_op(conv1, "conv2", 5, 5, 32)   # 20*20*64
    p1 = tf.layers.flatten(conv2)
    p2 = tf.layers.dense(inputs=p1, units=1024, activation=tf.nn.relu)
    p4 = tf.layers.dense(inputs=p2, units=10, activation=None)
    print("P4-shape===", p4.shape)
    y_pre = tf.arg_max(tf.nn.softmax(logits=p4), 1, name="y_label")
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=lable, logits=p4))
    num = tf.equal(y_pre, tf.arg_max(lable, 1))
    correct_prediction = tf.reduce_mean(tf.cast(num, tf.float32))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    return train_step, cross_entropy, correct_prediction, p4, y_pre




if __name__ == "__main__":

    batch_size = 1024
    num_epochs = 512
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    y = tf.placeholder(tf.float32, shape=[None, 10], name="y")

    train_step, cross_entropy, acc_num, p4, y_pre = LENet(x, y)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver(max_to_keep=1)
        for i in range(0, num_epochs):
            data_x, data_y = mnist.train.next_batch(batch_size)
            test_x, test_y = mnist.test.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: data_x, y: data_y})
            if (i+1) % 5 == 0:
                loss = sess.run(cross_entropy, feed_dict={x: data_x, y: data_y})
                print("epoch=%s,loss=%s" % (i, loss))
                acc = sess.run(acc_num, feed_dict={x: test_x, y: test_y})
                print("epoch=%s,acc_entropy=%s" % (i, acc))

        saver.save(sess, 'model/mnist.ckpt', global_step=num_epochs)











