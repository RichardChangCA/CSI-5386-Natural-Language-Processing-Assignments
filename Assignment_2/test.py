import numpy as np

predictions = [2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
labels = [1, 1, 1, 0, 1, 2, 1, 1, 1, 0, 2, 1, 0, 2, 1, 1, 2, 0, 0, 2, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 2, 0, 2, 1, 2, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 2, 1, 1, 1, 1, 0, 2, 1, 1, 1, 0, 0, 2, 0, 2, 1, 2, 0, 0, 0, 0, 1, 1, 0, 2, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 2, 2, 1, 1, 0, 1, 2, 0, 0, 2, 1, 1, 1, 1, 1, 0, 0]

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

print(evaluation_calculation(predictions,labels))