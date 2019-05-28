import tensorflow as tf
from bertmaster import modeling, tokenization
import os
from bertpreprocess import convert_single_example, next_batch
import numpy as np



#MAX_SQE_LENGTH = 64

# 这里是下载下来的bert配置文件
bert_config = modeling.BertConfig.from_json_file("bert_model/uncased_L-12_H-768_A-12/bert_config.json")
#  创建bert的输入
#input_ids=tf.placeholder (shape=[64,128],dtype=tf.int32,name="input_ids")
#input_mask=tf.placeholder (shape=[64,128],dtype=tf.int32,name="input_mask")
#segment_ids=tf.placeholder (shape=[64,128],dtype=tf.int32,name="segment_ids")
input_ids=tf.placeholder (shape=[None,None],dtype=tf.int32,name="input_ids")
input_mask=tf.placeholder (shape=[None,None],dtype=tf.int32,name="input_mask")
segment_ids=tf.placeholder (shape=[None,None],dtype=tf.int32,name="segment_ids")

# 创建bert模型
model = modeling.BertModel(
                           config=bert_config,
                           is_training=True,
                           input_ids=input_ids,
                           input_mask=input_mask,
                           token_type_ids=segment_ids,
                           use_one_hot_embeddings=False # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
                           )

#bert模型参数初始化的地方
init_checkpoint = "bert_model/uncased_L-12_H-768_A-12/bert_model.ckpt"
use_tpu = False
# 获取模型中所有的训练参数。
tvars = tf.trainable_variables()
# 加载BERT模型

(assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                           init_checkpoint)

tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

# 获取最后一层和倒数第二层。
encoder_last_layer = model.get_sequence_output()
#encoder_last2_layer = model.all_encoder_layers[-2]
#output_layer = model.get_pooled_output() # 这个获取句子的output

tf.logging.info("**** Trainable Variables ****")
# 打印加载模型的参数
for var in tvars:
    init_string = ""
    if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                    init_string)



def batch_for_lstm(batch_size, support_from, refute_from, support_line_number, refute_line_number, MAX_SQE_LENGTH):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #token = tokenization.CharTokenizer(vocab_file=bert_vocab_file)
        vocabload = tokenization.load_vocab(vocab_file="bert_model/uncased_L-12_H-768_A-12/vocab.txt")
        token = tokenization.FullTokenizer(vocab_file="bert_model/uncased_L-12_H-768_A-12/vocab.txt")
        query = "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."
        compare = None #"Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."
        #word_ids,word_mask,word_segment_ids = convert_single_example(MAX_SQE_LENGTH, token, vocabload, query, text_b=compare)
        
        x_batch, y_batch, new_support_from, new_refute_from = next_batch(batch_size, support_from, refute_from, MAX_SQE_LENGTH, support_line_number, refute_line_number)
        
        #    split_tokens = token.tokenize(query)
        #    word_ids = tokenization.convert_tokens_to_ids(vocabload, split_tokens)
        #    word_mask = [1] * len(word_ids)
        #    word_segment_ids = [0] * len(word_ids)
        #sess.run(train_step, feed_dict={input_ids:input_ids_single, input_mask:input_mask_single, segment_ids:segment_ids_single})
        #last, last2, output = sess.run([encoder_last_layer, encoder_last_layer, output_layer], feed_dict={input_ids:x_batch['word_ids'], input_mask:x_batch['word_mask'], segment_ids:x_batch['word_segment_id']})
        x_batch_transformed = []
        for i in range(len(y_batch)):
            last = sess.run(encoder_last_layer, feed_dict={input_ids:[x_batch['word_ids'][i]], input_mask:[x_batch['word_mask'][i]], segment_ids:[x_batch['word_segment_id'][i]]})
            x_batch_transformed.append(last[0])
        np_x_trans = np.array(x_batch_transformed)
        #print(y_batch)
        for_npy = []
        for label in y_batch:
            if (label[0] > 0.0):
                for_npy.append([1.0, 0.0])
                #supports
            else:
                for_npy.append([0.0, 1.0])
                #refutes
                        
                
        np_y = np.array(for_npy)
        #print(np_y)
        #print(np_y)
        #print(np_x_trans.shape)
        #print(np_y.shape)
        #print(np_x_trans)
        #print(np_y)
        #    last = sess.run(encoder_last_layer, feed_dict={input_ids:x_batch['word_ids'], input_mask:x_batch['word_mask'], segment_ids:x_batch['word_segment_id']})
        #    print(last)
        #print('last shape:{}'.format(last.shape))
        #print('last value:{}'.format(last))
        #output_layer = model.get_sequence_output()# 这个获取每个token的output 输出[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个
        #print(output_layer)
        #output_layer = model.get_pooled_output() # 这个获取句子的output
        print("data transfered.")
        return (np_x_trans, np_y, new_support_from, new_refute_from)


if __name__ == "__main__":
    batch_for_lstm(3, 2, 2, 73535, 28221, 64)



