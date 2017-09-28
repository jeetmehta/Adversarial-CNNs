# Import relevant modules
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Load input data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Start Tensorflow session & initialize variables
sess = tf.InteractiveSession()

# Initialize placeholders for input and ground truths/labels
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Initialize weights/biases for softmax model
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())

# Initialize model
y = tf.matmul(x, W) + b

# Define cross-entropy loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Train model
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for _ in range(1000):
	batch = mnist.train.next_batch(100)
	train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Evaluate model
correct_predicted = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predicted, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))