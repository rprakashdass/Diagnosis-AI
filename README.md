# Diagnosis-AI
This project is a simple binary classification task using a breast cancer dataset. It uses TensorFlow and Keras to train a neural network to predict whether a tumor is benign or malignant based on various features. The model achieved an accuracy of 93.86% on the test set.

#We import the pandas library to read the dataset from a CSV file using the read_csv() function. We read the dataset into a dataframe and store it in the variable dataset.

Next, we split the dataset into two parts, x and y. x represents the independent variables, i.e., all the columns except the target column. y represents the dependent variable, i.e., the target column.

We then split the dataset into training and testing sets using the train_test_split() function from the sklearn library. We split the data into 80% training set and 20% testing set. The training set is used to train the model, and the testing set is used to evaluate the performance of the model.

We then create a sequential model using the Keras library. A sequential model is a linear stack of layers, where we can add one layer at a time. We add three layers to the model: an input layer, a hidden layer, and an output layer. The input layer takes the shape of the training data, and the output layer has one neuron with a sigmoid activation function since this is a binary classification problem.

We compile the model using the compile() function, specifying the optimizer and loss function. We use the Adam optimizer and binary cross-entropy as the loss function since this is a binary classification problem. We also specify accuracy as the evaluation metric.

We then fit the model on the training data using the fit() function, specifying the number of epochs to train the model. We evaluate the model using the evaluate() function on the testing data.

The output of the evaluate() function provides the loss and accuracy of the model on the testing data. In our case, we get a loss of 0.172 and an accuracy of 0.938 on the testing data. This means that the model has an accuracy of 93.8% in predicting the diagnosis of breast cancer based on the input features.
