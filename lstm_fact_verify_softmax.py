import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
from berttransformersoftmax import batch_for_lstm

#mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

batch_size = 3
support_from = 50000
refute_from = 50000
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


    for epoch in range(3000):
        for batch in range(n_batch):
            batch_xs, batch_ys, support_from, refute_from = batch_for_lstm(200, support_from, refute_from, support_line_number, refute_line_number, MAX_SQE_LENGTH)
            print(batch_xs.shape)
            sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
            #######
        testbatch_xs, testbatch_ys, support_from, refute_from = batch_for_lstm(10, support_from, refute_from, support_line_number, refute_line_number, MAX_SQE_LENGTH)
        print(testbatch_xs[0])
        predrnn, pred, acc = sess.run([prediction, predicted, accuracy], feed_dict={x:testbatch_xs, y:testbatch_ys})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
        print(predrnn)
        print(pred)
        saver.save(sess, 'net/factverifysoftmax.ckpt')



