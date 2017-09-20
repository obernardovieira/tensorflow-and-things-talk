import tensorflow as tf

def checkit(g):
    return [[g[0],g[1]],[g[2],g[3]]]

x = tf.placeholder(tf.int64, shape=(2, 2))
y = tf.placeholder(tf.int64, shape=(2, 2))
z = tf.add(x, y)
tx = checkit(input('x')) #1,2,2,1
ty = checkit(input('y')) #2,1,1,2
sess = tf.Session()
print(sess.run(z, feed_dict={x: tx, y: ty}))
