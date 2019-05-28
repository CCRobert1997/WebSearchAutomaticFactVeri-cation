import json
import os
import nltk
import time
import math
from collections import Counter
import re
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
import time
print ("Start : %s" % time.ctime())

start = time.time()

from multiprocessing import Process

def parse_document(document):
   document = re.sub('\n', ' ', document)
   if isinstance(document, str):
       document = document
   else:
       raise ValueError('Document is not string!')
   document = document.strip()
   sentences = nltk.sent_tokenize(document)
   sentences = [sentence.strip() for sentence in sentences]
   return sentences


def search_final_sentence(wordlst):
    possible_iden=[]

    word1=' '.join(wordlst)
    ix = open_dir("index_ws_LZ_new")
    qp = QueryParser('page_title', schema=ix.schema)
    q = qp.parse(word1)
    with ix.searcher() as searcher:
        results = searcher.search(q, limit=50)
        for result in results:
            iden=(result.fields()['page_title']).replace(' ','_')+' '+result.fields()['sentence_numbers']
            possible_iden.append(iden)

    if len(possible_iden)==0:
        if len(wordlst) >= 2:
            for i in range(len(wordlst) - 1):
                word1 = wordlst[i]
                word2 = wordlst[i + 1]
                combine_word = word1 + ' ' + word2
                ix = open_dir("index_ws_LZ_new")
                qp = QueryParser('page_title', schema=ix.schema)
                q = qp.parse(combine_word)
                with ix.searcher() as searcher:
                    results = searcher.search(q, limit=20)
                    for result in results:
                        iden = (result.fields()['page_title']).replace(' ', '_') + ' ' + result.fields()[
                            'sentence_numbers']

                        possible_iden.append(iden)

    possible_iden=list(set(possible_iden))

    return possible_iden


def task1():
    f = open('test-unlabelled', encoding="utf-8-sig")
    js_test = json.load(f)
    claim_and_selected_sentence = {}

    for claim_id in js_test:
        claim = js_test[claim_id]['claim']
        if claim[-1] == '.':
            claim = claim[:-1]
        tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in parse_document(claim)]
        tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]
        key_word_for_search = []

        for word_tag in tagged_sentences[0]:
            if (not word_tag[0].islower()) and (word_tag[1] != 'DT'):
                key_word_for_search.append(word_tag[0])

        key_word_for_search = list(set(key_word_for_search))

        final_sentence_lst=search_final_sentence(key_word_for_search)

        claim = js_test[claim_id]['claim']

        claim_and_selected_sentence[claim]=final_sentence_lst

    with open('out_test_unlabel', "wb") as f:
        f.write((json.dumps(claim_and_selected_sentence, indent=2).encode("utf-8")))
        f.close()

task1()


