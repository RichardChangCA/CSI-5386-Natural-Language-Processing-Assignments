import tensorflow as tf
import ktrain
from ktrain import text as txt
import csv
import numpy as np

# BERT Bidirectional Encoder Representations from Transformers

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
        return sentences, labels


def model_training():
    x_train, y_train = extract_data("SICK_train.txt")

    x_test, y_test = extract_data("SICK_test_annotated.txt")
    class_names = ['ENTAILMENT', 'NEUTRAL', 'CONTRADICTION']
    (x_train,y_train), (x_test,y_test), preproc = txt.texts_from_array(x_train=x_train,y_train=y_train,
                                                                    x_test=x_test,y_test=y_test,
                                                                    class_names=class_names,
                                                                    preprocess_mode='bert',
                                                                    maxlen=60,max_features=35000)

    model = txt.text_classifier('bert',train_data=(x_train,y_train),preproc=preproc)
    learner = ktrain.get_learner(model, train_data=(x_train,y_train),batch_size=10)
    learner.fit_onecycle(2e-5,4)
    learner.validate(val_data=(x_test,y_test),class_names=class_names)

    predictor = ktrain.get_predictor(learner.model,preproc)

    # let's save the predictor for later use
    predictor.save('models_bert/bert_predictor')

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


def model_prediction():
    # reload the predictor
    reloaded_predictor = ktrain.load_predictor('models_bert/bert_predictor')

    x_test, y_test = extract_data("SICK_test_annotated.txt")

    y_test = np.array(y_test)
    prediction = reloaded_predictor.predict(x_test)
    prediction_labels = []
    for i in prediction:
        if i == 'ENTAILMENT':
            prediction_labels.append(0)
        elif i == 'NEUTRAL':
            prediction_labels.append(1)
        elif i == 'CONTRADICTION':
            prediction_labels.append(2)
    prediction_labels = np.array(prediction_labels)
    accuracy_sum,entailment_TP,entailment_TN,entailment_FP,entailment_FN,neutral_TP,neutral_TN,neutral_FP,neutral_FN,contradiction_TP,contradiction_TN,contradiction_FP,contradiction_FN = evaluation_calculation(prediction_labels,y_test)
    
    f_results = open("results_part_1_bert.txt",'w+')
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
