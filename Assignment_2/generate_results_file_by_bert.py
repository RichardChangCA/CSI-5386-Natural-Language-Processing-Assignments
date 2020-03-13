import tensorflow as tf
import ktrain
from ktrain import text as txt
import csv
import numpy as np
from sklearn import preprocessing

def extract_data(file_name):
    with open(file_name,"r") as data:
        train = csv.DictReader(data, delimiter='\t')
        sentences = []
        item_id = []
        for row in train:
            item_id.append(row["pair_ID"])
            sentences.append(row["sentence_A"]+". "+row["sentence_B"])
        return item_id,sentences

def result_generation():
    entailment_predictor = ktrain.load_predictor('models_bert/bert_predictor')

    relatedness_predictor = ktrain.load_predictor('models_bert/bert_regression_predictor')

    item_id,x_test = extract_data("SICK_test_annotated.txt")

    entailment_prediction = entailment_predictor.predict(x_test)

    relatedness_prediction = relatedness_predictor.predict(x_test)

    item_id=np.array(item_id)
    entailment_prediction=np.array(entailment_prediction)
    relatedness_prediction=np.array(relatedness_prediction)

    relatedness_prediction = preprocessing.MinMaxScaler(feature_range=(0, 5)).fit_transform(relatedness_prediction)
    relatedness_prediction = relatedness_prediction.flatten()
    relatedness_prediction=np.array(relatedness_prediction)

    array=[]
    result_file = open("Results.txt",'w')

    result_file.write("pair_id, entrailment_judgment, relatedness_score\n")
    for i in range(len(item_id)):
        result_file.write(str(item_id[i])+', '+str(entailment_prediction[i])+', '+str(relatedness_prediction[i])+'\n')
    
    result_file.close()


result_generation()