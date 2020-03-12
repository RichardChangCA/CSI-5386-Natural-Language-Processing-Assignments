# Stacking LSTM hidden layers makes the model deeper, more accurately earning the description as a deep learning technique.

import tensorflow as tf
import numpy as np
import csv
import time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Bidirectional, Dropout, Dense
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import models, layers


physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)


glove_vectors_file = '/home/lingfeng/web_data/embeddings/glove.6B/glove.6B.200d.txt'
# http://nlp.stanford.edu/data/glove.6B.zip
glove_wordmap = {}
with open(glove_vectors_file, "r") as glove:
    for line in glove:
        name, vector = tuple(line.split(" ", 1))
        glove_wordmap[name] = np.fromstring(vector, sep=" ")

#Constants setup
max_hypothesis_length, max_evidence_length = 30, 30
batch_size, vector_size, hidden_size = 128, 200, 64
lstm_size = hidden_size
training_iterations_count = 1000000
display_step = 100

def sentence2sequence(sentence):
    """
    - Turns an input sentence into an (n,d) matrix, 
        where n is the number of tokens in the sentence
        and d is the number of dimensions each word vector has.
    
      Tensorflow doesn't need to be used here, as simply
      turning the sentence into a sequence based off our 
      mapping does not need the computational power that
      Tensorflow provides. Normal Python suffices for this task.
    """
    tokens = sentence.lower().split(" ")
    rows = []
    words = []
    #Greedy search for tokens
    for token in tokens:
        i = len(token)
        while len(token) > 0 and i > 0:
            word = token[:i]
            if word in glove_wordmap:
                rows.append(glove_wordmap[word])
                words.append(word)
                token = token[i:]
                i = len(token)
            else:
                i = i-1
    return rows, words

def fit_to_size(matrix, shape):
    res = np.zeros(shape)
    slices = [slice(0,min(dim,shape[e])) for e, dim in enumerate(matrix.shape)]
    res[slices] = matrix[slices]
    return res

def split_data_into_scores(file_name):
    with open(file_name,"r") as data:
        train = csv.DictReader(data, delimiter='\t')
        evi_sentences = []
        hyp_sentences = []
        labels = []
        scores = []
        for row in train:
            hyp_sentences.append(np.vstack(
                    sentence2sequence(row["sentence_A"].lower())[0]))
            evi_sentences.append(np.vstack(
                    sentence2sequence(row["sentence_B"].lower())[0]))
            if row["entailment_judgment"] == 'ENTAILMENT':
                # labels.append(0)
                labels.append([1,0,0])
            elif row["entailment_judgment"] == 'NEUTRAL':
                # labels.append(1)
                labels.append([0,1,0])
            elif row["entailment_judgment"] == 'CONTRADICTION':
                # labels.append(2)
                labels.append([0,0,1])
            else:
                assert row["entailment_judgment"] == "INVALID"
            scores.append(row["relatedness_score"])
        
        hyp_sentences = np.stack([fit_to_size(x, (max_hypothesis_length, vector_size))
                          for x in hyp_sentences])
        evi_sentences = np.stack([fit_to_size(x, (max_evidence_length, vector_size))
                          for x in evi_sentences])          
        return hyp_sentences, evi_sentences, labels, np.array(scores)

def model_training():
    hyp_sentences, evi_sentences, correct_values, correct_scores = split_data_into_scores("SICK_train.txt")

    x_train = np.concatenate((hyp_sentences, evi_sentences), 1)
    y_train = np.array(correct_values)

    hyp_sentences, evi_sentences, correct_values, correct_scores = split_data_into_scores("SICK_test_annotated.txt")

    x_test = np.concatenate((hyp_sentences, evi_sentences), 1)
    y_test = np.array(correct_values)

    dropout_prop = 0.3
    batch_size, vector_size, hidden_size = 128, 200, 64
    lstm_size = hidden_size

    model = models.Sequential()
    model.add(Bidirectional(LSTM(units=lstm_size,return_sequences=True)))
    model.add(Dropout(dropout_prop))
    model.add(Bidirectional(LSTM(units=lstm_size,return_sequences=True)))
    model.add(Dropout(dropout_prop))
    model.add(Bidirectional(LSTM(units=lstm_size)))
    model.add(Dropout(dropout_prop))

    model.add(Dense(3,activation='sigmoid'))

    model.compile('adam','categorical_crossentropy',metrics=['accuracy'])

    model.fit(x_train,y_train,batch_size=batch_size,epochs=20,validation_data=[x_test,y_test])
    model.summary()
    model.save("models_stacked_lstm/models_stacked_lstm.h5")

def confusion_values(predictions,labels):

    # true positive, true negative, false positive, false negative
    # 0 is majority, 1 is minority
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    total_num = len(predictions)

    for i in range(total_num):
        if labels[i] == 1:
            if predictions[i] == 1:
                TP += 1
            else:
                FN += 1
        else:
            if predictions[i] == 1:
                FP += 1
            else:
                TN += 1
    return TP,TN,FP,FN

def evaluation_calculation(predictions,labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    numbers = len(predictions)
    accuracy_sum = 0
    predictions_entailment = []
    labels_entailment = []
    predictions_neutral = []
    labels_neutral = []
    predictions_contradiction = []
    labels_contradiction = []
    for i in range(numbers):
        if(predictions[i] == labels[i]):
            accuracy_sum += 1

        if(predictions[i] == 0):
            predictions_entailment.append(1)
            predictions_neutral.append(0)
            predictions_contradiction.append(0)
        elif(predictions[i] == 1):
            predictions_entailment.append(0)
            predictions_neutral.append(1)
            predictions_contradiction.append(0)
        elif(predictions[i] == 2):
            predictions_entailment.append(0)
            predictions_neutral.append(0)
            predictions_contradiction.append(1)

        if(labels[i] == 0):
            labels_entailment.append(1)
            labels_neutral.append(0)
            labels_contradiction.append(0)
        elif(labels[i] == 1):
            labels_entailment.append(0)
            labels_neutral.append(1)
            labels_contradiction.append(0)
        elif(labels[i] == 2):
            labels_entailment.append(0)
            labels_neutral.append(0)
            labels_contradiction.append(1)

    entailment_TP,entailment_TN,entailment_FP,entailment_FN = confusion_values(predictions_entailment,labels_entailment)
    neutral_TP,neutral_TN,neutral_FP,neutral_FN = confusion_values(predictions_neutral,labels_neutral)
    contradiction_TP,contradiction_TN,contradiction_FP,contradiction_FN = confusion_values(predictions_contradiction,labels_contradiction)
    return accuracy_sum,entailment_TP,entailment_TN,entailment_FP,entailment_FN,neutral_TP,neutral_TN,neutral_FP,neutral_FN,contradiction_TP,contradiction_TN,contradiction_FP,contradiction_FN


def output_label(item):
    one_hot = 0 
    max_value = item[0]
    if(item[1] > max_value):
        one_hot = 1
        max_value = item[1]
    if(item[2] > max_value):
        one_hot = 2

    if(one_hot == 0):
        return np.array([1,0,0])
    elif(one_hot == 1):
        return np.array([0,1,0])
    else:
        return np.array([0,0,1])

def model_prediction():
    
    with open("SICK_test_annotated.txt","r") as data:
        test = csv.DictReader(data, delimiter='\t')
        evi_sentences = []
        hyp_sentences = []
        labels = []
        for row in test:
            hyp_sentences.append(np.vstack(
                    sentence2sequence(row["sentence_A"].lower())[0]))
            evi_sentences.append(np.vstack(
                    sentence2sequence(row["sentence_B"].lower())[0]))
            if row["entailment_judgment"] == 'ENTAILMENT':
                labels.append(0)
            elif row["entailment_judgment"] == 'NEUTRAL':
                labels.append(1)
            elif row["entailment_judgment"] == 'CONTRADICTION':
                labels.append(2)
        
        hyp_sentences = np.stack([fit_to_size(x, (max_hypothesis_length, vector_size))
                          for x in hyp_sentences])
        evi_sentences = np.stack([fit_to_size(x, (max_evidence_length, vector_size))
                          for x in evi_sentences])

    
    x = np.concatenate((hyp_sentences, evi_sentences), 1)
    labels = np.array(labels)

    model = models.load_model("models_stacked_lstm/models_stacked_lstm.h5")

    prediction = model.predict_classes(x)

    prediction = np.array(prediction)

    accuracy_sum,entailment_TP,entailment_TN,entailment_FP,entailment_FN,neutral_TP,neutral_TN,neutral_FP,neutral_FN,contradiction_TP,contradiction_TN,contradiction_FP,contradiction_FN = evaluation_calculation(prediction,labels)
    
    f_results = open("results_part_1_stacked_lstm.txt",'w+')
    f_results.write("accuracy: "+str(accuracy_sum/len(prediction))+"\n")
    f_results.write("entailment_TP: "+str(entailment_TP)+"\n")
    f_results.write("entailment_TN: "+str(entailment_TN)+"\n")
    f_results.write("entailment_FP: "+str(entailment_FP)+"\n")
    f_results.write("entailment_FN: "+str(entailment_FN)+"\n")
    f_results.write("neutral_TP: "+str(neutral_TP)+"\n")
    f_results.write("neutral_TN: "+str(neutral_TN)+"\n")
    f_results.write("neutral_FP: "+str(neutral_FP)+"\n")
    f_results.write("neutral_FN: "+str(neutral_FN)+"\n")
    f_results.write("contradiction_TP: "+str(contradiction_TP)+"\n")
    f_results.write("contradiction_TN: "+str(contradiction_TN)+"\n")
    f_results.write("contradiction_FP: "+str(contradiction_FP)+"\n")
    f_results.write("contradiction_FN: "+str(contradiction_FN)+"\n")

    f_results.close()

def main():
    # model_training()
    model_prediction()

if __name__ == '__main__':
    main()