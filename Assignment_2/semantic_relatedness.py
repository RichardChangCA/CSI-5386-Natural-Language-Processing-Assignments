import tensorflow as tf
import numpy as np
import csv
import time
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import mean_squared_error

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
learning_rate = 0.1
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
            scores.append([float(row["relatedness_score"])])
        
        hyp_sentences = np.stack([fit_to_size(x, (max_hypothesis_length, vector_size))
                          for x in hyp_sentences])
        evi_sentences = np.stack([fit_to_size(x, (max_evidence_length, vector_size))
                          for x in evi_sentences])          
        return (hyp_sentences, evi_sentences), labels, np.array(scores)

def model_training():
    data_feature_list, _, correct_scores = split_data_into_scores("SICK_train.txt")

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
    y = tf.placeholder(tf.float32, [N,1], 'score')
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
    fc_weight = tf.get_variable('fc_weight', [2*hidden_size,1], 
                                initializer = fc_initializer)
    # fc_weight: Storage for the fully connected layer's weights.
    fc_bias = tf.get_variable('bias',[1])
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
    with tf.variable_scope('pearson_correlation'):
        predicts = tf.cast(classification_scores, 'float32')
        y_label = tf.cast(y, 'float32')
        predicts = tf.reshape(predicts,[N])
        y_label = tf.reshape(y_label,[N])
        pearson_correlation, update_op = tf.contrib.metrics.streaming_pearson_correlation(predicts,y_label)
        # pearson_correlation = tf.reduce_mean(tf.cast(corr, tf.float32))

    with tf.variable_scope("loss"):
        # mse = (y-classification_scores)**2
        # loss = tf.reduce_mean(mse)
        loss = tf.compat.v1.losses.mean_squared_error(y,classification_scores)
        total_loss = loss + weight_decay * tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    opt_op = optimizer.minimize(total_loss)
    # Initialize variables
    # init = tf.global_variables_initializer()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

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
                        correct_scores[batch])
        
        # Run the optimization with these initialized values
        sess.run([opt_op], feed_dict={hyp: hyps, evi: evis, y: ys})
        sess.run(update_op,feed_dict={hyp: hyps, evi: evis, y: ys})
        # display_step: how often the accuracy and loss should 
        #   be tested and displayed.
        if (i/batch_size) % display_step == 0:
            # Calculate batch accuracy
            pearson = sess.run(pearson_correlation, feed_dict={hyp: hyps, evi: evis, y: ys})
            # Calculate batch loss
            tmp_loss = sess.run(loss, feed_dict={hyp: hyps, evi: evis, y: ys})
            # Display results
            print("Iter " + str(i/batch_size) + ", Minibatch Loss= " + \
                "{:.6f}".format(tmp_loss) + ", Training pearson correlation= " + \
                "{:.5f}".format(pearson))
    saver.save(sess,"models/model_relatedness.ckpt")
    sess.close()

# def softmax(x): 
#     e_x = np.exp(x - np.max(x)) 
#     return e_x / e_x.sum(axis=0) 

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
            scores.append([float(row["relatedness_score"])])
        
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
    y = tf.placeholder(tf.float32, [N,1], 'score')

    lstm_back = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    lstm_drop_back = tf.contrib.rnn.DropoutWrapper(lstm_back, input_p, output_p)

    fc_weight = tf.get_variable('fc_weight', [2*hidden_size,1])
    # fc_weight: Storage for the fully connected layer's weights.
    fc_bias = tf.get_variable('bias',[1])

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
        saver.restore(sess,"models/model_relatedness.ckpt")
        test_batch_num= int(len(hyp_sentences)/128)
        prediction_all = np.array([])
        for batch in range(test_batch_num):
            prediction_labels = []
            hyps, evis = (hyp_sentences[batch*128:(batch+1)*128,:],
                        evi_sentences[batch*128:(batch+1)*128])
            prediction = sess.run(classification_scores, feed_dict={hyp: hyps,
                                                            evi: evis})
            prediction_all = np.concatenate((prediction_all,np.array(prediction).flatten()))
        # print(np.array(scores[:test_batch_num*128]).flatten())
        # print(prediction_all)
        f_results_2 = open("results_part_2.txt",'w')
        f_results_2.write("\npearson_correlation: "+ str(pearsonr(np.array(scores[:test_batch_num*128]).flatten(),prediction_all)))
        f_results_2.write("\nmean squared error: "+ str(mean_squared_error(np.array(scores[:test_batch_num*128]).flatten(),prediction_all)))
        f_results_2.write("\nspearman correlation: "+ str(spearmanr(np.array(scores[:test_batch_num*128]).flatten(),prediction_all)))
        f_results_2.close()
    # total_items = test_batch_num*128

def main():
    model_training()
    model_prediction()

if __name__ == '__main__':
    main()