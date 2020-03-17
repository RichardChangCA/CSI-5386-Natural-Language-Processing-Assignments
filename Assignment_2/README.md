Instruction of running source codes:

Import dataset SICK_test.txt, SICK_test_annotated.txt, SICK_train.txt and SICK_trial.txt in the same folder with source codes.

Import glove pre-trained word-embedding files in the correct path.

Create folder named as: models, models_stacke_lstm, models_bert

In TensorFlow 1.x version:

python3 text_entailment.py

python3 semantic_relatedness.py

In TensorFlow 2.x version:

Installing ktrain package: pip3 install ktrain

python3 text_entailment_bert.py

python3 semantic_relatedness_bert.py

python3 text_entailment_stacked_lstm.py

python3 semantic_relatedness_stacked_lstm.py

All results can be generated as .txt files.

Files Instruction:

models folder stores Bi-LSTM trained model

models_stacked_lstm folder stores Stacked Bi-LSTM trained model

Bert model is so huge that can not be stroed in GitHub easily, and training bert model is time-consuming.

Results.txt stores pair-id, predicted class and predicted scroes

SICK_test.txt, SICK_test_annotated.txt, SICK_train.txt and SICK_trial.txt are datasets

generate_results_file_by_bert.py is used to generate Results.txt file, because we can know bert perform the best between these models after comparing trained models.

results_part_1.txt stores the confusion matrix(True Positive, True Negative, False Positive, False Negative in one-vs-others method) and accuracy of Bi-LSTM model.

results_part_1_stacked_lstm.txt and results_part_1_bert.txt stroes the same thing as results_part_1.txt

results_part_2.txt stroes pearson_correlation, mean squared error and spearman correlation

results_part_2_stacked_lstm.txt and results_part_2_bert.txt stroes the same thing as results_part_2.txt

semantic_relatedness.py, semantic_relatedness_bert.py and semantic_relatedness_stacked_lstm.py are the source codes for task 2.

stacked_bidirectional_lstm_keras_model.txt stores the structure of stacked Bi-LSTM model.

text_entailment.py, text_entailment_bert.py and text_entailment_stacked_lstm.py are the sources codes of task 1.

scripy.py in the calculate_metric folder caculates precision, recall and f-measure based on confusion matrices.



