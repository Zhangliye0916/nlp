# -*-coding:utf-8-*-
import tensorflow as tf
import warnings

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    # message = tf.constant('Welcome to the exciting world of Deep Neural Networks!')
    # with tf.Session() as session:
    #     print(session.run(message).decode())

    v1 = tf.constant([1, 2, 3])
    v2 = tf.constant([2, 3, 4])
    v3 = tf.add(v1, v2)

    with tf.Session() as sess:
        print(sess.run([v1,v2,v3]))

    t1 = tf.zeros([2,3], tf.int32)
    t2 = tf.ones_like(t1)
    t3 = tf.add(t1,t2)
    with tf.Session() as sess:
        print(sess.run(t3))

    t4 = tf.linspace(1.0, 10.0, 10, "odd")
    with tf.Session() as sess:
        print(sess.run(t4))

    t5 = tf.random_normal([2, 3], 0, 1.0, tf.float16)
    t6 = tf.Variable(t5)

    with tf.Session() as sess:
        print(sess.run(t6.initial_value))