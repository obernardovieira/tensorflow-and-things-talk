import tensorflow as tf
from PIL import Image

def resize_image(url):
    # open image
    col = Image.open(url)
    # resize image
    col = col.resize((28, 28), Image.ANTIALIAS)
    # make it black and white
    gray = col.convert('L')
    bw = gray.point(lambda x: 0 if x<128 else 255, '1')
    # get transformed data
    pixels = list(bw.getdata())
    width, height = bw.size
    #
    pixels = [[1. if pixels[g] == 0 else 0. for g in xrange(28 * 28)]]
    return pixels

# Parameters
model_path = "/tmp/model.ckpt"

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# Running first session
print("Starting 1st session...")
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, model_path)
    print("Model restored")

    c = sess.run([tf.argmax(pred, 1)], feed_dict={x: resize_image("image.jpg")})
    print(c)