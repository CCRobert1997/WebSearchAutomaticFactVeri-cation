
import json
import os
from multiprocessing import Pool
import time

print ("Start : %s" % time.ctime())

with open('train.json','r') as f:
    js_train=json.load(f)

refute_title=[]
for i in js_train:
    templen=len(js_train[i]['evidence'])
    if js_train[i]['label']=='REFUTES':
        for evi_num in range(templen):
            refute_title.append(js_train[i]['evidence'][evi_num][0]+' '+str(js_train[i]['evidence'][evi_num][1]))


path="./wiki_text"
def task1():
    with open('lzx/refute_lines.txt', 'w', encoding='UTF-8') as wf:
        os.chdir(path)
        list_of_text = os.listdir()
        for i in list_of_text:
            with open(i, 'r', encoding='UTF-8') as f:
                for line in f:
                    line_split = line.split(" ")
                    line_title = line_split[0] + ' ' + str(line_split[1])
                    for t in temp_refute_title_1:
                        if line_title == t:
                            wf.writelines(line)
                            break
							
for i in js_train:
    templen=len(js_train[i]['evidence'])
    if js_train[i]['label']=='SUPPORTS':
        for evi_num in range(templen):
            support_title.append(js_train[i]['evidence'][evi_num][0]+' '+str(js_train[i]['evidence'][evi_num][1]))


path="./wiki_text"
def task1():
    with open('lzx/support_lines.txt', 'w', encoding='UTF-8') as wf:
        os.chdir(path)
        list_of_text = os.listdir()
        for i in list_of_text:
            with open(i, 'r', encoding='UTF-8') as f:
                for line in f:
                    line_split = line.split(" ")
                    line_title = line_split[0] + ' ' + str(line_split[1])
                    for t in temp_refute_title_1:
                        if line_title == t:
                            wf.writelines(line)
                            break



 
