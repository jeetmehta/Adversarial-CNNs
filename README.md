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

## Fooling MNIST: This Project

Having understood the surrounding context behind the problem, I can now present the specific application that this project is tackling. 

The basic premise of the project is two-fold: First, create a classifier that can successfully classify the popular handwriting data set, [MNIST](http://yann.lecun.com/exdb/mnist/). This will be done using a Deep Convolutional Network, with the exact same structure and parameters as shown in this [TensorFlow CNN Tutorial](https://www.tensorflow.org/get_started/mnist/pros#deep-mnist-for-experts). 

Next, after having trained and evaluated the performance of the classifier (this one performed at **99.2%** after having trained on 20,000 training samples), generate some adversarial images that are specifically created to force the network to **misclassify a "2" as a "6".** More specifically, a set of 10 "2" digits will be taken from the dataset, and passed through this generation process such that the network will consistently misclassify each one as a "6". 

How these adversarial images can be generated is what will be covered in the next section.

## Generating Adversarial Images

Since Ian Goodfellow's original paper on the topic (definitely a must read for those interested: [Explaining and Harnessing Adversarial Examples](https://arxiv.org/pdf/1412.6572.pdf)), there have been a variety of different methods suggested in following papers to generate adversarial images. However, the original method suggested by Goodfellow works well as a base for this project, and is explained below.

### Fast Gradient Sign Method

Let **θ** be the parameters of a model, **x** the input to the model, **y** the targets associated with x (for machine learning tasks that have targets) and **J(θ, x, y)** be the cost used to train the neural network.

We can linearize the cost function around the current value of θ, obtaining an optimal max-norm constrained perturbation of:

			η = sign (∇x J(θ, x, y)).

Since the gradient can be calculated effectively using back-propagation, this method of calculating and obtaining a perturbation signal is a **single-step process**, and thus is both **efficient, and fast.**

The issue with this approach is that the **misclassified target class is random.** There is no control over what the misclassified/resulting output of the network will be once the noise been added to the image. This is where my current implementation comes in.

### Implementation: Targeted Iterative Fast Gradient Sign

There are essentially two issues/problems that the originally identified approach contains:

 1. Random target class
 2. Single step approach (leads to non-minimal loss)

Directing the misclassification by the network towards a specific target class can be done by using the **target label instead of the ground truth** in the equation given above. This basically leads to the calculation of cross-entropy loss between the target class, and the noisy input image, thereby producing a directed gradient.

However, a single gradient step may not be sufficient to converge towards the target class. Larger *epsilon/noise weight* values will not minimize the cross-entropy loss between the target class and the input, and force convergence on the wrong output. Smaller values require multiple iterations, and thus a single-step will not suffice in this case either. This was identified empirically personally during the development of the algorithm as well as in research, where it led to misclassification in only 63-69% of inputs.

The iterative alternative, proposed originally by Kurakin et al. (see [Adversarial Examples in the Physical World](https://arxiv.org/pdf/1607.02533.pdf)) and summarized by Goodfellow in his most recent paper on the topic ([Adversarial Machine Learning At Scale](https://arxiv.org/pdf/1611.01236.pdf)) works as follows:

![The iterative fast gradient sign method as proposed by Kurakin and Goodfellow](https://lh3.googleusercontent.com/-Jn90kNBBbC4/WdG5udtcXYI/AAAAAAAAY0w/tYahAVGN_-cSq9F28ZfDj-q8WG0_JqfXACLcBGAs/s0/Screen+Shot+2017-10-01+at+11.58.56+PM.png "Screen Shot 2017-10-01 at 11.58.56 PM.png")

This was the approach utilized within this project, and the related output is shown in the next section.

## Results

The iterative, targeted fast gradient sign method (FGSM) approach worked amazingly well, allowing a consistent misclassification of the input "2"'s as "6"'s, thereby meeting the goals of the project.

For visualization purposes, the resulting output images are plotted and shown below.

![Network output showing the effects of adversarial perturbations](https://lh3.googleusercontent.com/-SAdMnhO7Hp4/WdG8L7zUK4I/AAAAAAAAY1M/7AAt8bSMZwErqO062k4IQnOGm6sCbUpOwCLcBGAs/s0/output.png "output.png")

## References

Numerous research papers, blogs and articles were used for this project, and links to all of them are provided below:

 1. https://www.tensorflow.org/get_started/mnist/pros#deep-mnist-for-experts
 2. http://karpathy.github.io/2015/03/30/breaking-convnets/
 3. https://arxiv.org/pdf/1412.6572.pdf
 4. http://www.evolvingai.org/files/DNNsEasilyFooled_cvpr15.pdf
 5. https://openreview.net/pdf?id=SJCscQcge
 6. https://arxiv.org/abs/1707.04131
 7. https://arxiv.org/pdf/1702.04267.pdf
 8. https://arxiv.org/pdf/1511.04599v1.pdf
 9. http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture16.pdf
 10. https://github.com/tensorflow/cleverhans
 11. https://github.com/Evolving-AI-Lab/fooling
 12. https://arxiv.org/pdf/1611.01236.pdf
 13. http://www.anishathalye.com/2017/07/25/synthesizing-adversarial-examples/