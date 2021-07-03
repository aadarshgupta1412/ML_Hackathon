# Handwritten expression evaluator

AUTHORS: Vishal Saini, Kuldeep Meena, Aadarsh Gupta

The software, based on the architectures of Image-to-Text, aims to recognize basic handwritten expressions and evaluate & find the results, meanwhile also differentiating on the basis of position of operator, i.e. whether the handwritten expression is infix, prefix or postfix

Techniques and Datasets used:
- CNN(Convolution Neural Network): Convolution Neural Network has been used to train for the classification of the image parts to corresponding digits or operators
- MNIST and self created dataset: MNIST dataset has been used for training the model to recognize digits and self created dataset consisting 25k test images have been used for training the model to recognize operators 

### Subtask-1 (inference1.py)
- Loads the trained model
- Reads images from testImage folder
- Classifies test images into prefix/infix/postfix
- Saves the predicted category to csv file

### Subtask-2 (inference2.py)
- Loads the trained model
- Reads images from testImage folder
- Evaluates the corresponding mathematical result of the expression given in the image
- Saves the predicted value to csv file

### Instructions For Using The Software:
-   Unzip the directory to your desired location.
-   Run command pip install -r requirements.txt in terminal
-   Replace file with name "test_data"(line no. 44 and 39 repectively) in the code of inference1.py and inference2.py to the desired test folder name 
-   Run inference1.py by using command python inference1.py and the output will be a new csv file with name AIMLC_HackTheSummer_1.csv 
-   Run inference2.py by using command python inference1.py and the output will be a new csv file with name AIMLC_HackTheSummer_2.csv

