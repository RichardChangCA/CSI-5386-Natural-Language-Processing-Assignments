# -*- coding: utf-8 -*-
"""public-20newsgroups-BERT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1AH3fkKiEqBpVpO5ua00scp7zcHs5IDLK
"""

# install ktrain on Google Colab
# !pip3 install ktrain

# import ktrain and the ktrain.text modules
import ktrain
from ktrain import text
import numpy as np
ktrain.__version__

"""# Multiclass Text Classification Using BERT and Keras
In this example, we will use ***ktrain*** ([a lightweight wrapper around Keras](https://github.com/amaiya/ktrain)) to build a model using the dataset employed in the **scikit-learn** tutorial: [Working with Text Data](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html).  As in the tutorial, we will sample 4 newsgroups to create a relatively small multiclass text classification dataset.  The objective is to accurately classify each document into one of these four newsgroup topic categories.  This will provide us an opportunity to see **BERT** in action on a relatively smaller training set.  Let's fetch the [20newsgroups dataset ](http://qwone.com/~jason/20Newsgroups/) using scikit-learn.
"""

# fetch the dataset using scikit-learn
categories = ['alt.atheism', 'soc.religion.christian',
             'comp.graphics', 'sci.med']
from sklearn.datasets import fetch_20newsgroups
train_b = fetch_20newsgroups(subset='train',
   categories=categories, shuffle=True, random_state=42)
test_b = fetch_20newsgroups(subset='test',
   categories=categories, shuffle=True, random_state=42)

print('size of training set: %s' % (len(train_b['data'])))
print('size of validation set: %s' % (len(test_b['data'])))
print('classes: %s' % (train_b.target_names))

x_train = train_b.data
y_train = train_b.target

x_test = test_b.data
y_test = test_b.target

"""## STEP 1:  Load and Preprocess the Data
Preprocess the data using the `texts_from_array function` (since the data resides in an array).
If your documents are stored in folders or a CSV file you can use the `texts_from_folder` or `texts_from_csv` functions, respectively.
"""

(x_train,  y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=x_train, y_train=y_train,
                                                                       x_test=x_test, y_test=y_test,
                                                                       class_names=train_b.target_names,
                                                                       preprocess_mode='bert',
                                                                       maxlen=350, 
                                                                       max_features=35000)

"""## STEP 2:  Load the BERT Model and Instantiate a Learner object"""

# you can disregard the deprecation warnings arising from using Keras 2.2.4 with TensorFlow 1.14.
model = text.text_classifier('bert', train_data=(x_train, y_train), preproc=preproc)
learner = ktrain.get_learner(model, train_data=(x_train, y_train), batch_size=6)

"""## STEP 3: Train the Model

We train using one of the three learning rates recommended in the BERT paper: *5e-5*, *3e-5*, or *2e-5*.
Alternatively, the ktrain Learning Rate Finder can be used to find a good learning rate by invoking `learner.lr_find()` and `learner.lr_plot()`, prior to training.
The `learner.fit_onecycle` method employs a [1cycle learning rate policy](https://arxiv.org/pdf/1803.09820.pdf).
"""

learner.fit_onecycle(2e-5, 4)

"""We can use the `learner.validate` method to test our model against the validation set.
As we can see, BERT achieves a **96%** accuracy, which is quite a bit higher than the 91% accuracy achieved by SVM in the [scikit-learn tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html).
"""

learner.validate(val_data=(x_test, y_test), class_names=train_b.target_names)

"""## How to Use Our Trained BERT Model

We can call the `learner.get_predictor` method to obtain a Predictor object capable of making predictions on new raw data.
"""

predictor = ktrain.get_predictor(learner.model, preproc)

predictor.get_classes()

predictor.predict(test_b.data[0:1])

# we can visually verify that our prediction of 'sci.med' for this document is correct
print(test_b.data[0])

# we predicted the correct label
print(test_b.target_names[test_b.target[0]])

"""The `predictor.save` and `ktrain.load_predictor` methods can be used to save the Predictor object to disk and reload it at a later time to make predictions on new data."""

# let's save the predictor for later use
predictor.save('/tmp/my_predictor')

# reload the predictor
reloaded_predictor = ktrain.load_predictor('/tmp/my_predictor')

# make a prediction on the same document to verify it still works
reloaded_predictor.predict(test_b.data[0:1])