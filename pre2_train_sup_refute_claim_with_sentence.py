import json
import os
import nltk
import time

print ("Start : %s" % time.ctime())
start = time.time()
os.chdir("C:/Users/lenovo/PycharmProjects/untitled/wikijson")


f=open('train.json',encoding="utf-8-sig")
js_train=json.load(f)

refute_dict=[]

time=0
for i in js_train:
    time+=1
    print(time)
    if js_train[i]['label']=='REFUTES':
        temp_line = []
        templen=len(js_train[i]['evidence'])
        for evi_num in range(templen):
            evi_title=js_train[i]['evidence'][evi_num][0]+' '+str(js_train[i]['evidence'][evi_num][1])
            with open('lzx/refute_lines.txt', 'r', encoding='UTF-8') as rf:
                for line in rf:
                    line_split = line.split(" ")
                    line_title = line_split[0] + ' ' + str(line_split[1])
                    if line not in temp_line and evi_title==line_title:
                        temp_line.append(line)
                        print(line)
        refute_dict[js_train[i]['claim']]=temp_line

with open('train_refute_with_sentence.txt', 'w', encoding='UTF-8') as wf:
    wf.write(json.dumps(refute_dict))
	
for i in js_train:
    time+=1
    print(time)
    if js_train[i]['label']=='SUPPORTS':
        temp_line = []
        templen=len(js_train[i]['evidence'])
        for evi_num in range(templen):
            evi_title=js_train[i]['evidence'][evi_num][0]+' '+str(js_train[i]['evidence'][evi_num][1])
            with open('lzx/support_lines.txt', 'r', encoding='UTF-8') as rf:
                for line in rf:
                    line_split = line.split(" ")
                    line_title = line_split[0] + ' ' + str(line_split[1])
                    if line not in temp_line and evi_title==line_title:
                        temp_line.append(line)
                        print(line)
        refute_dict[js_train[i]['claim']]=temp_line

with open('lzx/train_support_with_sentence.txt', 'w', encoding='UTF-8') as wf:
    wf.write(json.dumps(refute_dict))
