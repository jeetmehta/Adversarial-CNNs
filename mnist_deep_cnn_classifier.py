# Import relevant modules
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Load input data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Define placeholders for input and ground truths/labels
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Initializes weights with small positive values (normally distributed with std deviation of 0.1)
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

# Initializes biases with small positive values (normally distributed with std deviation of 0.1)
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# Convolutional operation: zero padding to ensure input_size = output_size, and a stride of 1
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# Pooling operation: max pooling, 2x2 kernel blocks, stride of 1, zero padding to ensure input_size = output_size
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Create first convolutional layer - 32 filters, each one is 5x5 -> produces 32 outputs
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# Reshape input into 4D before putting it through the layers: 28x28 image, 1 color channel
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Apply first layer: X -> Conv_1 -> ReLu -> Pool_1
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Create second convolutional layer - 64 filters, each one is 5x5 -> produces 64 outputs
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

# Apply second layer
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected third layer - has 1024 neurons to process the entire image
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# Reshape input before application
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

# Apply FC third layer
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Apply dropout layer to reduce overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# Define cross-entropy loss function and training configuration
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Start Tensorflow sesssion
with tf.Session() as sess:

	# Initialize variables
	sess.run(tf.global_variables_initializer())

	# Train model
	for _ in range(1000):
		batch = mnist.train.next_batch(50)
		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# Evaluate model
correct_predicted = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predicted, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))