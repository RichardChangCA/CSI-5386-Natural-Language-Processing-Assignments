# Stacking LSTM hidden layers makes the model deeper, more accurately earning the description as a deep learning technique.

import tensorflow as tf
import numpy as np
import csv
import time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Bidirectional, Dropout, Dense
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras import models, layers
import tensorflow.python.keras.backend as K
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_squared_error

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
            scores.append(float(row["relatedness_score"]))
        
        hyp_sentences = np.stack([fit_to_size(x, (max_hypothesis_length, vector_size))
                          for x in hyp_sentences])
        evi_sentences = np.stack([fit_to_size(x, (max_evidence_length, vector_size))
                          for x in evi_sentences])          
        return hyp_sentences, evi_sentences, labels, np.array(scores)

# def pearson_correlation_metric(y_true,y_predict):
#     return K.variable(np.array(pearsonr(K.eval(y_true),K.eval(y_predict)))[0])
#     # K.constant

def pearson_correlation_metric(y_true, y_pred):
    #normalise
    n_y_true = (y_true - K.mean(y_true[:])) / K.std(y_true[:])
    n_y_pred = (y_pred - K.mean(y_pred[:])) / K.std(y_pred[:])  

    top=K.sum((n_y_true[:]-K.mean(n_y_true[:]))*(n_y_pred[:]-K.mean(n_y_pred[:])),axis=[-1,-2])
    bottom=K.sqrt(K.sum(K.pow((n_y_true[:]-K.mean(n_y_true[:])),2),axis=[-1,-2])*K.sum(K.pow(n_y_pred[:]-K.mean(n_y_pred[:]),2),axis=[-1,-2]))

    result=top/bottom

    return K.mean(result)

def model_training():
    hyp_sentences, evi_sentences, correct_values, correct_scores = split_data_into_scores("SICK_train.txt")

    x_train = np.concatenate((hyp_sentences, evi_sentences), 1)
    y_train = correct_scores

    hyp_sentences, evi_sentences, correct_values, correct_scores = split_data_into_scores("SICK_test_annotated.txt")

    x_test = np.concatenate((hyp_sentences, evi_sentences), 1)
    y_test = correct_scores

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

    model.add(Dense(1,activation='relu'))

    # model.compile('adam','mse',metrics=['mae'])
    model.compile('sgd','mse',metrics=[pearson_correlation_metric])

    model.fit(x_train,y_train,batch_size=batch_size,epochs=100,validation_data=[x_test,y_test])
    # with open("stacked_bidirectional_lstm_keras_model.txt",'w+') as fn:
    #     model.summary(print_fn=lambda x: fn.write(x + '\n'))
    model.save("models_stacked_lstm/models_stacked_lstm_relatedness.h5")


def model_prediction():
    
    with open("SICK_test_annotated.txt","r") as data:
        test = csv.DictReader(data, delimiter='\t')
        evi_sentences = []
        hyp_sentences = []
        scores = []
        for row in test:
            hyp_sentences.append(np.vstack(
                    sentence2sequence(row["sentence_A"].lower())[0]))
            evi_sentences.append(np.vstack(
                    sentence2sequence(row["sentence_B"].lower())[0]))

            scores.append(float(row["relatedness_score"]))
        
        hyp_sentences = np.stack([fit_to_size(x, (max_hypothesis_length, vector_size))
                          for x in hyp_sentences])
        evi_sentences = np.stack([fit_to_size(x, (max_evidence_length, vector_size))
                          for x in evi_sentences])

    
    x = np.concatenate((hyp_sentences, evi_sentences), 1)
    scores = np.array(scores)

    dependencies = {
        'pearson_correlation_metric': pearson_correlation_metric
    }
    model = models.load_model("models_stacked_lstm/models_stacked_lstm_relatedness.h5",custom_objects=dependencies)

    prediction = model.predict(x)

    prediction = np.array(prediction)

    prediction = prediction.flatten()

    f_results_2 = open("results_part_2_stacked_lstm.txt",'w')
    f_results_2.write("\npearson_correlation: "+ str(pearsonr(scores,prediction)))
    f_results_2.write("\nmean squared error: "+ str(mean_squared_error(scores,prediction)))
    f_results_2.write("\nspearman correlation: "+ str(spearmanr(scores,prediction)))
    f_results_2.close()

    # print(prediction)

def main():
    # model_training()
    model_prediction()

if __name__ == '__main__':
    main()