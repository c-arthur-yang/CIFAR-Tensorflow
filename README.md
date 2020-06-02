CIFAR-10 is a dataset for Object Recognition in Images. Although it was closed 6 years before, we still can use it as a heuristic exercise for our CV model.

Previously, people can load the training set and test set directly from cifar10 set in keras. But in this competition, what we have are two image sets for training and testing respectively, a trainLabels for the training set. In this case, our fist step is to decode the images in both sets by using tf.iamge.decode_png(), which returns a tensor for each image.
