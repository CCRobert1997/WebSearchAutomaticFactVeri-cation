import json
import os
import nltk
import time
from collections import Counter
print ("Start : %s" % time.ctime())
start = time.time()


def task1():
    count = 0
    f=open('test-unlabelled.json',encoding="utf-8-sig")
    js_train=json.load(f)
    identifier_to_complete_sentence={}
    for claim in js_train:
        count+=1
        complete_sent=[]
        for file_identifier in js_train[claim]:
            with open('Testlabel_with_useful_line.txt', 'r', encoding='UTF-8') as f:
                for line in f:
                    line_split = line.split(" ")
                    first_word = line_split[0]
                    line_num=line_split[1]
                    wiki_iden=first_word+' '+line_num
                    if wiki_iden==file_identifier:
                        complete_sent.append(line)
                        break
        print(count)
        identifier_to_complete_sentence[claim]=complete_sent

    with open('combine.json', "wb") as f:
        f.write((json.dumps(identifier_to_complete_sentence, indent=2).encode("utf-8")))
        f.close()

task1()


