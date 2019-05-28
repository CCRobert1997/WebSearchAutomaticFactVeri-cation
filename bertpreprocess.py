import tensorflow as tf
from bertmaster import modeling, tokenization
import os
import json
import linecache

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
        
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_single_example(max_seq_length, tokenizer, vocab, text_a, text_b=None):
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenizer.tokenize(text_b)# 这里主要是将中文分字
    if tokens_b:
        # 如果有第二个句子，那么两个句子的总长度要小于 max_seq_length - 3
        # 因为要为句子补上[CLS], [SEP], [SEP]
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # 如果只有一个句子，只用在前后加上[CLS], [SEP] 所以句子长度要小于 max_seq_length - 3
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]
            
            # 转换成bert的输入，注意下面的type_ids 在源码中对应的是 segment_ids
            # (a) 两个句子:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            # (b) 单个句子:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0     0   0   0  0     0 0
            #
            # 这里 "type_ids" 主要用于区分第一个第二个句子。
            # 第一个句子为0，第二个句子是1。在预训练的时候会添加到单词的的向量中，但这个不是必须的
            # 因为[SEP] 已经区分了第一个句子和第二个句子。但type_ids 会让学习变的简单
            
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
    #input_ids = tokenizer.convert_tokens_to_ids(tokens)# 将中文转换成ids
    input_ids = tokenization.convert_tokens_to_ids(vocab, tokens)# 将中文转换成ids
    # 创建mask
    input_mask = [1] * len(input_ids)
    # 对于输入进行补0
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return input_ids,input_mask,segment_ids # 对应的就是创建bert模型时候的input_ids,input_mask,segment_ids 参数




def next_batch(batch_size, support_from, refute_from, max_seq_length, support_line_number, refute_line_number):
    vocabload = tokenization.load_vocab(vocab_file="bert_model/uncased_L-12_H-768_A-12/vocab.txt")
    token = tokenization.FullTokenizer(vocab_file="bert_model/uncased_L-12_H-768_A-12/vocab.txt")
    new_support_from = support_from - 1
    new_refute_from = refute_from - 1
    #    print(new_support_from)
    #    print(new_refute_from)
    x_batch = {'word_ids': [], 'word_mask': [], 'word_segment_id': []}
    y_batch = []
    for i in range(batch_size):
        testnextline = True
        while (testnextline):
            try:
                new_support_from = (new_support_from + 1)%support_line_number
                new_refute_from = (new_refute_from + 1)%refute_line_number
                supportline = json.loads(linecache.getline('lzx/lineshift_support_claim.json', (new_support_from)%support_line_number))
                
                refuteline = json.loads(linecache.getline('lzx/lineshift_refute_claim.json', (new_refute_from)%refute_line_number))
                
                testnextline = False
            except:
                new_support_from = (support_from + 1)%support_line_number
                new_refute_from = (refute_from + 1)%support_line_number
        #print(list(supportline.keys())[0])
        
        for sentence in list(supportline.values())[0]:
            word_id,word_mask,word_segment_id = convert_single_example(max_seq_length, token, vocabload, list(supportline.keys())[0], text_b=sentence)
            x_batch['word_ids'].append(word_id)
            x_batch['word_mask'].append(word_mask)
            x_batch['word_segment_id'].append(word_segment_id)
            y_batch.append([1.])
        for sentence in list(refuteline.values())[0]:
            word_id,word_mask,word_segment_id = convert_single_example(max_seq_length, token, vocabload, list(refuteline.keys())[0], text_b=sentence)
            x_batch['word_ids'].append(word_id)
            x_batch['word_mask'].append(word_mask)
            x_batch['word_segment_id'].append(word_segment_id)
            y_batch.append([0.])


        #        print(list(refuteline.keys())[0])
        #        for sentence in list(refuteline.values())[0]:
        #            print(sentence)
    print("data loaded.")
    return (x_batch, y_batch, new_support_from, new_refute_from)
if __name__ == "__main__":
    support_from_i = 50000
    refute_from_i = 20000
    print(next_batch(3, support_from_i, refute_from_i, 128, 73535, 28221))
    #print(next_batch(10, support_from_i, refute_from_i, 128))


