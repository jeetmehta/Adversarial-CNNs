# Import relevant modules
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load input data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Class that describes the deep CNN: contains all relevant variables and functions
class DeepCNN:

	def __init__(self, learning_rate=1e-4, batch_size=50, training_size=20000, save_file_name='deep_cnn_model'):
		self.save_filename = save_file_name
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.training_size = training_size

	# Initializes weights with small positive values (normally distributed with std deviation of 0.1)
	def init_weight_variables(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	# Initializes biases with small positive values (normally distributed with std deviation of 0.1)
	def init_bias_variables(self, shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	# Convolutional operation: zero padding to ensure input_size = output_size, and a stride of 1
	def conv2d(self, x, W):
		return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

	# Pooling operation: max pooling, 2x2 kernel blocks, stride of 1, zero padding to ensure input_size = output_size
	def max_pool_2x2(self, x):
		return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	# Builds the CNN: Structure is: (FC -> ReLu -> Pool) * 2 -> FC -> DO -> RD
	def build_network(self, x, keep_prob):

		# Create first convolutional layer - 32 filters, each one is 5x5 -> produces 32 outputs
		W_conv1 = self.init_weight_variables([5, 5, 1, 32])
		b_conv1 = self.init_bias_variables([32])

		# Reshape input into 4D before putting it through the layers: 28x28 image, 1 color channel
		x_image = tf.reshape(x, [-1, 28, 28, 1])

		# Apply first layer: X -> Conv_1 -> ReLu -> Pool_1
		h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
		h_pool1 = self.max_pool_2x2(h_conv1)

		# Create second convolutional layer - 64 filters, each one is 5x5 -> produces 64 outputs
		W_conv2 = self.init_weight_variables([5,5,32,64])
		b_conv2 = self.init_bias_variables([64])

		# Apply second layer
		h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = self.max_pool_2x2(h_conv2)

		# Fully connected third layer - has 1024 neurons to process the entire image
		W_fc1 = self.init_weight_variables([7*7*64, 1024])
		b_fc1 = self.init_bias_variables([1024])

		# Reshape input before application
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

		# Apply FC third layer
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		# Apply dropout layer to reduce overfitting
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

		# Readout layer
		W_fc2 = self.init_weight_variables([1024, 10])
		b_fc2 = self.init_bias_variables([10])
		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

		# Initialize saver
		self.saver = tf.train.Saver()

		return y_conv


	# Trains the CNN using cross-entropy loss 
	def train_network(self, x, y_conv, y_, keep_prob, save_model):

		# Define cross-entropy loss function, gradient and training configuration
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
		gradient = tf.gradients(cross_entropy, x)
		train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)

		# Start Tensorflow sesssion
		with tf.Session() as sess:

			# Initialize variables
			sess.run(tf.global_variables_initializer())

			# Train model
			for _ in range(self.training_size):
				batch = mnist.train.next_batch(self.batch_size)
				train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

			# Save model
			if save_model:
				self.saver.save(sess, self.save_filename)


	# Runs/evaluates/predicts using the trained model, given/at a specific input
	def predict_network(self, x, input_data, y_conv, y_, keep_prob):

		# Start Tensorflow sesssion
		with tf.Session() as sess:

			# Load the trained model
			self.saver.restore(sess, self.save_filename)

			# Run model on given input
			sess.run(y_conv, feed_dict = {x: input_data, keep_prob: 1.0})
			correct_predicted = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_predicted, tf.float32))
			print(accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

	# Evaluates and prints the performance of network, given the ground truth labels
	def evaluate_network(self, x, y_conv, y_, keep_prob):

		# Start Tensorflow sesssion
		with tf.Session() as sess:

			# Load the trained model
			self.saver.restore(sess, self.save_filename)

			# Evaluate model
			correct_predicted = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_predicted, tf.float32))
			print(accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

	# Creates adversarial examples using a linear combination "2" digit inputs and noise
	def create_adv_samples(self, x, y_conv, y_, keep_prob, noise, sample_size):

		# Start Tensorflow sesssion
		with tf.Session() as sess:

			# Load the trained model
			self.saver.restore(sess, self.save_filename)

			# Declare target class: "6" in this case
			target_one_hot = np.reshape([0, 0, 0, 0, 0, 0, 1., 0, 0, 0], (-1, 10))

			# Looping indices/variables
			idx, num_count = 0, 0

			# Storage variables for later visualization
			original_images = []
			pertubations = []
			adversarial_images = []

			# Generate 10 adversarial examples
			while (num_count < 10):

				# Process the digit only if it's a "2"
				if (mnist.train.labels[idx][2] == 1):

					# Store image locally and for visualization
					image = np.reshape(mnist.train.images[idx], (-1, 784))
					original_images.append(mnist.train.images[idx])

					# Calculate loss and gradients
					cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
					grad = tf.gradients(cross_entropy, x)
					np_grad = sess.run(grad, feed_dict = {x: image, y_: target_one_hot, keep_prob: 1.0})
					signed_grad = np.sign(np_grad[0])

					# Iterative Target Class: Fast Gradient Sign Method
					adv_x = tf.convert_to_tensor(image)
					alpha = 2
					eps = 0.25
					learning_rate = 1e-4
					num_iterations = 10

					# Iteratively use the gradient to move the adversarial image towards the target class
					for k in range(0, num_iterations):

						# Gradient descent
						cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
						optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
						loss_value = sess.run([optim_step, cross_entropy], feed_dict={x: image, y_: target_one_hot, keep_prob: 1.0})[1]

						# Clip function
						above = image + eps
						below = image - eps
						kernel = alpha * signed_grad

						# Store pertubation
						pertubations.append(kernel)

						# Update adversarial image
						adv_x = tf.clip_by_value(adv_x - kernel, below, above)
						
						# Output iteration # and loss
						print('step %d, loss=%g' % (k+1, loss_value))

					# Feed adversarial image to network and predict
					adv_x = np.reshape(adv_x.eval(), (-1, 784))
					pred2 = sess.run(y_conv, feed_dict = {x: adv_x, y_: target_one_hot, keep_prob: 1.0})
					label2 = np.argmax(pred2)
					print label2

					# Store output images
					adversarial_images.append(adv_x)

					num_count = num_count + 1
				idx = idx + 1

		output = [original_images, pertubations, adversarial_images]
		return output

# Plots the input image sets of originals, pertubations, and resulting adversarials
def visualize_images(image_set):

	# Store inputs
	originals = image_set[0]
	pertubations = image_set[1]
	adversarials = image_set[2]

	# Define plot structure
	f, ax_arr = plt.subplots(10,3)

	cols = ['Original Image', 'Delta', 'Adversarial Image']
	for ax, col in zip(ax_arr[0], cols):
	    ax.set_title(col)

	for i in range(0, len(originals)):

		# Show original image
		original_image = originals[i]
		original_image = original_image.reshape(28, 28)
		ax_arr[i, 0].imshow(original_image)
		ax_arr[i, 0].axis('off')

		# Show pertubation
		pertubation = pertubations[i]
		pertubation = pertubation.reshape(28, 28)
		ax_arr[i, 1].imshow(pertubation)
		ax_arr[i, 1].axis('off')

		# Show adversarial output
		adversarial_image = adversarials[i]
		adversarial_image = adversarial_image.reshape(28, 28)
		ax_arr[i, 2].imshow(adversarial_image)
		ax_arr[i, 2].axis('off')

	plt.show()


# Main function
def main():

	# Define placeholders for input, ground truths & dropout probabilities
	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])
	keep_prob = tf.placeholder(tf.float32)

	# Initialize & build CNN model -> save output variable
	model = DeepCNN(1e-4, 50, 20000, 'deep_cnn_model')
	y_conv = model.build_network(x, keep_prob)

	# Train the network
	# model.train_network(x, y_conv, y_, keep_prob, True)

	# Evaluate performance
	# model.evaluate_network(x, y_conv, y_, keep_prob)

	# Predict performance
	# model.predict_network(x, mnist.test.images, y_conv, y_, keep_prob)

	# Generate adversarial examples
	output_set = model.create_adv_samples(x, y_conv, y_, keep_prob, 0.10, 10)

	# Vizualize output
	visualize_images(output_set)

# Runs main function only if the script is explicitly called -> prevents it from running during imports
if __name__ == "__main__":
	main()