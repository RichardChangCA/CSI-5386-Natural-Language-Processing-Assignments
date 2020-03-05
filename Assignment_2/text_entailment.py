import tensorflow as tf
import numpy as np
import csv
import time

# Reference: https://github.com/Steven-Hewitt/Entailment-with-Tensorflow

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
weight_decay = 0.0001
learning_rate = 0.01
input_p, output_p = 0.5, 0.5 # dropout keep probability
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
                # labels.append([0])
                labels.append([1,0,0])
            elif row["entailment_judgment"] == 'NEUTRAL':
                # labels.append([1])
                labels.append([0,1,0])
            elif row["entailment_judgment"] == 'CONTRADICTION':
                # labels.append([2])
                labels.append([0,0,1])
            else:
                assert row["entailment_judgment"] == "INVALID"
            scores.append(row["relatedness_score"])
        
        hyp_sentences = np.stack([fit_to_size(x, (max_hypothesis_length, vector_size))
                          for x in hyp_sentences])
        evi_sentences = np.stack([fit_to_size(x, (max_evidence_length, vector_size))
                          for x in evi_sentences])          
        return (hyp_sentences, evi_sentences), labels, np.array(scores)

def model_training():
    data_feature_list, correct_values, correct_scores = split_data_into_scores("SICK_train.txt")

    correct_values = np.array(correct_values)

    l_h, l_e = max_hypothesis_length, max_evidence_length
    N, D, H = batch_size, vector_size, hidden_size
    l_seq = l_h + l_e

    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    lstm_drop =  tf.contrib.rnn.DropoutWrapper(lstm, input_p, output_p)

    # N: The number of elements in each of our batches, 
    #   which we use to train subsets of data for efficiency's sake.
    # l_h: The maximum length of a hypothesis, or the second sentence.  This is
    #   used because training an RNN is extraordinarily difficult without 
    #   rolling it out to a fixed length.
    # l_e: The maximum length of evidence, the first sentence.  This is used
    #   because training an RNN is extraordinarily difficult without 
    #   rolling it out to a fixed length.
    # D: The size of our used GloVe or other vectors.
    hyp = tf.placeholder(tf.float32, [N, l_h, D], 'hypothesis')
    evi = tf.placeholder(tf.float32, [N, l_e, D], 'evidence')
    y = tf.placeholder(tf.float32, [N, 3], 'label')
    # hyp: Where the hypotheses will be stored during training.
    # evi: Where the evidences will be stored during training.
    # y: Where correct scores will be stored during training.

    # lstm_size: the size of the gates in the LSTM, 
    #    as in the first LSTM layer's initialization.
    lstm_back = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    # lstm_back:  The LSTM used for looking backwards 
    #   through the sentences, similar to lstm.

    # input_p: the probability that inputs to the LSTM will be retained at each
    #   iteration of dropout.
    # output_p: the probability that outputs from the LSTM will be retained at 
    #   each iteration of dropout.
    lstm_drop_back = tf.contrib.rnn.DropoutWrapper(lstm_back, input_p, output_p)
    # lstm_drop_back:  A dropout wrapper for lstm_back, like lstm_drop.


    fc_initializer = tf.random_normal_initializer(stddev=0.1) 
    # fc_initializer: initial values for the fully connected layer's weights.
    # hidden_size: the size of the outputs from each lstm layer.  
    #   Multiplied by 2 to account for the two LSTMs.
    fc_weight = tf.get_variable('fc_weight', [2*hidden_size, 3], 
                                initializer = fc_initializer)
    # fc_weight: Storage for the fully connected layer's weights.
    fc_bias = tf.get_variable('bias', [3])
    # fc_bias: Storage for the fully connected layer's bias.

    # tf.GraphKeys.REGULARIZATION_LOSSES:  A key to a collection in the graph
    #   designated for losses due to regularization.
    #   In this case, this portion of loss is regularization on the weights
    #   for the fully connected layer.
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 
                        tf.nn.l2_loss(fc_weight)) 

    x = tf.concat([hyp, evi], 1) # N, (Lh+Le), d
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2]) # (Le+Lh), N, d
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, vector_size]) # (Le+Lh)*N, d
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, l_seq,)

    # x: the inputs to the bidirectional_rnn


    # tf.contrib.rnn.static_bidirectional_rnn: Runs the input through
    #   two recurrent networks, one that runs the inputs forward and one
    #   that runs the inputs in reversed order, combining the outputs.
    rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm, lstm_back,
                                                                x, dtype=tf.float32)
    # rnn_outputs: the list of LSTM outputs, as a list. 
    #   What we want is the latest output, rnn_outputs[-1]

    classification_scores = tf.matmul(rnn_outputs[-1], fc_weight) + fc_bias
    # The scores are relative certainties for how likely the output matches
    #   a certain entailment: 
    #     0: Positive entailment
    #     1: Neutral entailment
    #     2: Negative entailment
    with tf.variable_scope('Accuracy'):
        predicts = tf.cast(tf.argmax(classification_scores, 1), 'int32')
        y_label = tf.cast(tf.argmax(y, 1), 'int32')
        corrects = tf.equal(predicts, y_label)
        num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
        accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

    with tf.variable_scope("loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits = classification_scores, labels = y)
        loss = tf.reduce_mean(cross_entropy)
        total_loss = loss + weight_decay * tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    opt_op = optimizer.minimize(total_loss)
    # Initialize variables
    init = tf.global_variables_initializer()

    # Use TQDM if installed
    tqdm_installed = False
    try:
        from tqdm import tqdm
        tqdm_installed = True
    except:
        pass

    saver = tf.train.Saver()
    # Launch the Tensorflow session
    sess = tf.Session()
    sess.run(init)

    # training_iterations_count: The number of data pieces to train on in total
    # batch_size: The number of data pieces per batch
    training_iterations = range(0,training_iterations_count,batch_size)
    if tqdm_installed:
        # Add a progress bar if TQDM is installed
        training_iterations = tqdm(training_iterations)

    for i in training_iterations:

        # Select indices for a random data subset
        batch = np.random.randint(data_feature_list[0].shape[0], size=batch_size)
        
        # Use the selected subset indices to initialize the graph's 
        #   placeholder values
        hyps, evis, ys = (data_feature_list[0][batch,:],
                        data_feature_list[1][batch,:],
                        correct_values[batch])
        
        # Run the optimization with these initialized values
        sess.run([opt_op], feed_dict={hyp: hyps, evi: evis, y: ys})
        # display_step: how often the accuracy and loss should 
        #   be tested and displayed.
        if (i/batch_size) % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={hyp: hyps, evi: evis, y: ys})
            # Calculate batch loss
            tmp_loss = sess.run(loss, feed_dict={hyp: hyps, evi: evis, y: ys})
            # Display results
            print("Iter " + str(i/batch_size) + ", Minibatch Loss= " + \
                "{:.6f}".format(tmp_loss) + ", Training Accuracy= " + \
                "{:.5f}".format(acc))
    saver.save(sess,"models/model.ckpt")
    sess.close()

# def softmax(x): 
#     e_x = np.exp(x - np.max(x)) 
#     return e_x / e_x.sum(axis=0) 

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
    accuracy_sum,entailment_TP,entailment_TN,entailment_FP,entailment_FN,neutral_TP,neutral_TN,neutral_FP,neutral_FN,contradiction_TP,contradiction_TN,contradiction_FP,contradiction_FN=0,0,0,0,0,0,0,0,0,0,0,0,0

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

    l_h, l_e = max_hypothesis_length, max_evidence_length
    N, D, H = batch_size, vector_size, hidden_size
    l_seq = l_h + l_e

    tf.reset_default_graph()

    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    lstm_drop =  tf.contrib.rnn.DropoutWrapper(lstm, input_p, output_p)

    hyp = tf.placeholder(tf.float32, [N, l_h, D], 'hypothesis')
    evi = tf.placeholder(tf.float32, [N, l_e, D], 'evidence')
    y = tf.placeholder(tf.float32, [N, 3], 'label')

    lstm_back = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    lstm_drop_back = tf.contrib.rnn.DropoutWrapper(lstm_back, input_p, output_p)

    fc_weight = tf.get_variable('fc_weight', [2*hidden_size, 3])
    # fc_weight: Storage for the fully connected layer's weights.
    fc_bias = tf.get_variable('bias', [3])

    x = tf.concat([hyp, evi], 1) # N, (Lh+Le), d
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2]) # (Le+Lh), N, d
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, vector_size]) # (Le+Lh)*N, d
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, l_seq,)

    rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm, lstm_back,
                                                                x, dtype=tf.float32)

    classification_scores = tf.matmul(rnn_outputs[-1], fc_weight) + fc_bias

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        saver.restore(sess,"models/model.ckpt")
        test_batch_num= int(len(hyp_sentences)/128)
        for batch in range(test_batch_num):
            prediction_labels = []
            hyps, evis = (hyp_sentences[batch*128:(batch+1)*128,:],
                        evi_sentences[batch*128:(batch+1)*128])
            prediction = sess.run(classification_scores, feed_dict={hyp: hyps,
                                                            evi: evis})
            for i in prediction:
                prediction_labels.append(np.argmax(i))
            accuracy_sum_,entailment_TP_,entailment_TN_,entailment_FP_,entailment_FN_,neutral_TP_,neutral_TN_,neutral_FP_,neutral_FN_,contradiction_TP_,contradiction_TN_,contradiction_FP_,contradiction_FN_ = evaluation_calculation(prediction_labels,labels[batch*128:(batch+1)*128])
            accuracy_sum += accuracy_sum_
            entailment_TP += entailment_TP_
            entailment_TN += entailment_TN_
            entailment_FP += entailment_FP_
            entailment_FN += entailment_FN_
            neutral_TP += neutral_TP_
            neutral_TN += neutral_TN_
            neutral_FP += neutral_FP_
            neutral_FN += neutral_FN_
            contradiction_TP += contradiction_TP_
            contradiction_TN += contradiction_TN_
            contradiction_FP += contradiction_FP_
            contradiction_FN += contradiction_FN_
            # print(prediction_labels)
            # print(labels[batch*128:(batch+1)*128])
            # time.sleep(10)

        # left_batch = len(hyp_sentences) - test_batch_num*128
        # prediction_labels = []
        # hyps, evis = (hyp_sentences[test_batch_num*128:len(hyp_sentences),:],
        #             evi_sentences[test_batch_num*128:len(hyp_sentences)])
        # prediction = sess.run(classification_scores, feed_dict={hyp: hyps,
        #                                                 evi: evis})
        # for i in prediction:
        #     prediction_labels.append(np.argmax(i))
        # # print(prediction_labels)
        # # print(labels[test_batch_num*128:len(hyp_sentences)])
        # accuracy_sum_,entailment_TP_,entailment_TN_,entailment_FP_,entailment_FN_,neutral_TP_,neutral_TN_,neutral_FP_,neutral_FN_,contradiction_TP_,contradiction_TN_,contradiction_FP_,contradiction_FN_ = evaluation_calculation(prediction_labels,labels[test_batch_num*128:len(hyp_sentences)])
        # accuracy_sum += accuracy_sum_
        # entailment_TP += entailment_TP_
        # entailment_TN += entailment_TN_
        # entailment_FP += entailment_FP_
        # entailment_FN += entailment_FN_
        # neutral_TP += neutral_TP_
        # neutral_TN += neutral_TN_
        # neutral_FP += neutral_FP_
        # neutral_FN += neutral_FN_
        # contradiction_TP += contradiction_TP_
        # contradiction_TN += contradiction_TN_
        # contradiction_FP += contradiction_FP_
        # contradiction_FN += contradiction_FN_
        ######## ValueError: Cannot feed value of shape (63, 30, 200) for Tensor 'hypothesis:0', which has shape '(128, 30, 200)'

    total_items = test_batch_num*128
    f_results = open("results_part_1.txt",'w+')
    f_results.write("accuracy: "+str(accuracy_sum/total_items)+"\n")
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