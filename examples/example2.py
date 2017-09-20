import tensorflow as tf
x = [[1,2],[2,1]]
y = [[2,1],[1,2]]
z = tf.add(x, y)
sess = tf.Session()
print(sess.run(z))
