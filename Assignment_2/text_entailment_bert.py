# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# print("physical_devices-------------", len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

import ktrain
from ktrain import text as txt
import csv
import numpy as np

def extract_data(file_name):
    with open(file_name,"r") as data:
        train = csv.DictReader(data, delimiter='\t')
        sentences = []
        labels = []
        for row in train:
            sentences.append(row["sentence_A"]+". "+row["sentence_B"])
            # labels.append(row["entailment_judgment"])
            if row["entailment_judgment"] == 'ENTAILMENT':
                # labels.append([1,0,0])
                labels.append(0)
            elif row["entailment_judgment"] == 'NEUTRAL':
                # labels.append([0,1,0])
                labels.append(1)
            elif row["entailment_judgment"] == 'CONTRADICTION':
                # labels.append([0,0,1])
                labels.append(2)
        return np.array(sentences), np.array(labels)

x_train, y_train = extract_data("SICK_train.txt")
print(x_train.shape)
print(y_train.shape)
import time
time.sleep(10)
x_test, y_test = extract_data("SICK_test.txt")
class_names = ['ENTAILMENT', 'NEUTRAL', 'CONTRADICTION']
(x_train,y_train), (x_test,y_test), preproc = txt.texts_from_array(x_train=x_train,y_train=y_train,
                                                                   x_test=x_test,y_test=y_test,
                                                                   class_names=class_names,
                                                                   preprocess_mode='bert',
                                                                   maxlen=60,max_features=200)

model = txt.text_classifier('bert',train_data=(x_train,y_train),preproc=preproc)
learner = ktrain.get_learner(model, train_data=(x_train,y_train),batch_size=10)
learner.fit_onecycle(2e-5,4)
learner.validate(val_data=(x_test,y_test),class_names=class_names)

predictor = ktrain.get_predictor(learner.model,preproc)

prediction = predictor.predict(x_test)
