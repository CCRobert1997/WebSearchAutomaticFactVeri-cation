import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from berttransformersoftmax import batch_for_lstm

#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 3
support_from = 0
refute_from = 0
support_line_number = 73535
refute_line_number = 28221
MAX_SQE_LENGTH = 64

#n_inputs = 28 #one row has 28 data
#max_time = 28 #28 rows
lstm_size = 100 #hidden units
n_classes = 2
#batch_size = 20
n_batch = 9#mnist.train.num_examples // batch_size


#Create two place holders
x = tf.placeholder(tf.float32, [None,MAX_SQE_LENGTH,768])
y = tf.placeholder(tf.float32, [None,n_classes])

weights = tf.Variable(tf.truncated_normal([lstm_size,n_classes], stddev=0.1))
biases = tf.Variable(tf.constant(0.1,shape=[n_classes]))


def RNN(X,weights,biases):
    # inputs=[batch_size, max_time, n_inputs]
    inputs = X#tf.reshape(X,[-1,max_time,n_inputs])
    #define the LSTM basis CELL
    #lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(lstm_size)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    # final_state[0] is cell state
    # final_state[1] is hidden_state
    outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
    #results = tf.nn.softmax(tf.matmul(final_state[1],weights) + biases)
    results = tf.nn.softmax(tf.matmul(final_state[1],weights) + biases)
    return results


prediction = RNN(x, weights, biases)

#cross_entropy cost function
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#boolean list
#sigpred = tf.nn.sigmoid(prediction)
predicted = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1)) #argmax return the position of maximum
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
#
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
        saver.restore(sess, 'net/factverifysoftmax.ckpt')
        print("restore succeed")
    except:
        print("restore failed")

    from testpreprocess import data_for_lstm
    import json
    filepath = 'evidoneTop4.json'
    testbatch_xs, testbatch_ys, dict_index_range = data_for_lstm(filepath, MAX_SQE_LENGTH)

    predrnn = sess.run(prediction, feed_dict={x:testbatch_xs})


    dict_output = {}
    with open(filepath) as c_f:
        f_read = c_f.read()
        json_read = json.loads(f_read)
        for record in json_read.items():
            
            textlabel = "NOT ENOUGH INFO"
            highest_predict = 0.0
            evi_index = 0
            absolute_evi = 0
            for i in dict_index_range[record[0]]:
                if (predrnn[i][0] > predrnn[i][1]):
                    if (predrnn[i][0] > highest_predict):
                        highest_predict = predrnn[i][0]
                        textlabel = "SUPPORTS"
                        absolute_evi = evi_index
                else:
                    if (predrnn[i][1] > highest_predict):
                        highest_predict = predrnn[i][1]
                        textlabel = "REFUTES"
                        absolute_evi = evi_index
                evi_index += 1
            try:
                evilist = [record[1]["evidence"][absolute_evi][:-1]]
            except:
                evilist = []
                print(record)
            #for evi in record[1]["evidence"]:
                
            #    evilist.append(evi[:-1])
            dict_output[record[0]] = {"claim": record[1]["claim"], "label": textlabel, "evidence": evilist}
            #print(dict_output[record[0]])
    print("process done")
    with open('testoutput1top.json', "wb") as of:
        of.write((json.dumps(dict_output, indent=4).encode("utf-8")))
        print("file saved")



