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
        scores = []
        for row in train:
            sentences.append(row["sentence_A"]+". "+row["sentence_B"])
            scores.append(float(row["relatedness_score"]))
        return sentences, scores

def model_training():
    x_train, y_train = extract_data("SICK_train.txt")

    x_test, y_test = extract_data("SICK_test_annotated.txt")
    (x_train,y_train), (x_test,y_test), preproc = txt.texts_from_array(x_train=x_train,y_train=y_train,
                                                                    x_test=x_test,y_test=y_test,
                                                                    class_names = [],
                                                                    preprocess_mode='bert',
                                                                    maxlen=60,max_features=35000)
    # if class_names is empty, regression task is assumed.
    # metric: mae
    # loss function mse
    model = txt.text_regression_model('bert',train_data=(x_train,y_train),preproc=preproc)
    learner = ktrain.get_learner(model, train_data=(x_train,y_train),batch_size=10)
    learner.fit_onecycle(2e-5,20)

    predictor = ktrain.get_predictor(learner.model,preproc)

    # let's save the predictor for later use
    predictor.save('models_bert/bert_regression_predictor')

def model_prediction():
    # reload the predictor
    reloaded_predictor = ktrain.load_predictor('models_bert/bert_regression_predictor')

    prediction = predictor.predict(x_test)

    print(prediction)

def main():
    # model_training()
    model_prediction()

if __name__ == '__main__':
    main()