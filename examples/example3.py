import tensorflow as tf
x = tf.placeholder(tf.int64, shape=(2, 2))
y = tf.placeholder(tf.int64, shape=(2, 2))
z = tf.add(x, y)
tx = [[1,2],[2,1]]
ty = [[2,1],[1,2]]
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
print(sess.run(z, feed_dict={x: tx, y: ty}))
