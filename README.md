# Handwritten expression evaluator

AUTHORS: Vishal Saini,Kuldeep Meena,Aadarsh Gupta

The software aims to classify handwritten expressions into prefix, postfix, infix categories and finally evaluating the expression to produce desired value.The basic idea is to divide each image into 3 parts and classify each part to its corresponding meaning i.e a number 1 to 9 for an image with a number and +, - , * , / for the corresponding operators.

Techniques and Datasets used:
- CNN(Convolution Neural Network): Convolution Neural Network has been used to train for the classification of the image parts to corresponding digits or operators
- MNIST and self created dataset: MNIST dataset has been used for data used for training of digits and self created dataset consisting 25k test images have been used for training the model to recognize operators 
-
