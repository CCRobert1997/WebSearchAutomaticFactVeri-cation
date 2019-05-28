import json
import os
import nltk
import time
from collections import Counter
print ("Start : %s" % time.ctime())
start = time.time()

all_iden=[]

f = open('out_all_id_claim_with_iden.json', encoding="utf-8-sig")
js_one_of_1 = json.load(f)
for claim in js_one_of_1:
    for identifier in js_one_of_1[claim]:
        all_iden.append(identifier)

print(len(set(all_iden)))

final_iden=list(set(all_iden))

os.chdir('./')

with open('Testlabel_with_useful_line.txt','w',encoding='UTF-8') as wf:
    os.chdir('./wikitext')
    list_of_text = os.listdir()
    for wiki_text in list_of_text:
        print(wiki_text)
        with open(wiki_text, 'r', encoding='UTF-8') as f:
            for line in f:
                line_split=line.split(" ")
                line_title=line_split[0] + ' ' + str(line_split[1])
                if line_title in final_iden:
                    wf.writelines(line)
                    final_iden.remove(line_title)