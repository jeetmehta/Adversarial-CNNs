# CNNs & Adversarial Images - Fooling a MNIST Classifier

The goal of this project is to use TensorFlow to create a set of adversarial images, that can break/fool a CNN-based MNIST classifier. This analysis will first provide some background information regarding the use cases of adversarial pictures, and then go into the specific details of my implementation.

## Background Information

This section will provide some context into the impact of adversarial images on learning models, as well as cover the inherent source of the misclassification issue.

### The Problem


Deep convolutional networks (CNNs) are state-of-the-art models for image classification models and object detection. As the models become increasingly more innovative, and computing power continues to increase, they're often compared in popular media to possess "human-like" vision. However, these models posses some key limitations, such as misclassifying an image that's only slightly noisy with high confidence. Here's an example below:

![Example of the effect of gradient-based noise on classification systems](http://karpathy.github.io/assets/break/breakconv.png)

Clearly, the difference between the two images is so subtle that it's essentially impossible for humans to differentiate between them. Both images should be classified correctly as "panda". However, small amounts of noise like this make a **significant** impact on linear-learning based systems, such as Deep CNNs. This can be seen above, where the system classifies the noisy image as "gibbon", with **99.3 % confidence**

The same thing can be done in reverse, where noise is combined with small amounts of an actual image, and the classifier classifies it as the actual image with extremely high degrees of confidence. This can be seen below:

![Noise combined with small amounts of real world data produces a similar misclassification](http://karpathy.github.io/assets/break/break1.jpeg)

### Problem Source: Linear Nature of Learning Models

As Andrej Karpathy explains excellently in his blog (see [Breaking Linear Classifiers on ImageNet](http://karpathy.github.io/2015/03/30/breaking-convnets/)), the issue isn't specific to the images, CNNs, or even deep learning in general. The misclassification arises from the **linear nature** of these learning models. Consider CNNs, although they can model extremely non-linear functions, they are basically comprised of a bunch of linear components (each neuron just performs a linear weighted sum). Check out Karpathy's blog to understand more as to where this linearity comes from, and how it plays a role in the error. For the purposes of this project, it's just important to understand that the issue of misclassifying adversarial images is inherent to *all* linear-based models.

## Generating Adversarial Images

Since Ian Goodfellow's original paper on the topic (definitely a must read for those interested: [Explaining and Harnessing Adversarial Examples](https://arxiv.org/pdf/1412.6572.pdf)), there have been a variety of different methods suggested in following papers to generate adversarial images. However, the original method suggested by Goodfellow works well as a base for this project, and is explained below.

### Fast Gradient Sign Method

Let **θ** be the parameters of a model, **x** the input to the model, **y** the targets associated with x (for machine learning tasks that have targets) and **J(θ, x, y)** be the cost used to train the neural network.

We can linearize the cost function around the current value of θ, obtaining an optimal max-norm constrained perturbation of:

			η = sign (∇x J(θ, x, y)).

Since the gradient can be calculated effectively using back-propagation, this method of calculating and obtaining a perturbation signal is a **single-step process**, and thus is both **efficient, and fast.**

The issue with this approach is that the **misclassified target class is random.** There is no control over what the misclassified/resulting output of the network will be once the noise been added to the image. 