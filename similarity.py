import json
from gensimscore import doc_similarity



def sentence_split_head(text):
    seperate = ' '
    sentlist = text.split(seperate)
    compare = seperate.join(sentlist[2:])
    #head =  seperate.join(sentlist[:2])
    return (sentlist[0], sentlist[1], compare)

readfile = 'combine.json'
dict_output = {}
with open(readfile) as c_f:
    f_read = c_f.read()
    json_read = json.loads(f_read)

    for record in json_read.items():
        search = list(record[1].keys())[0]
        list_for_order = []
        index = 0
        scorelist = doc_similarity(list(record[1].values())[0], search, sys.argv[1])
        for sent_index in range(len(list(record[1].values())[0])):
            list_for_order.append((sent_index, scorelist[sent_index]))
        list_for_order.sort(key=lambda x:x[1], reverse=True)
        dict_output[record[0]] = {"claim": search, "label": "NOT ENOUGH INFO", "evidence": [["Soul_Food_-LRB-film-RRB-", 0, ]]}
        print(record[0])
        #print(search)
        try:
            head0_1, head1_1, sent1 = sentence_split_head(list(record[1].values())[0][list_for_order[0][0]])
            head0_2, head1_2, sent2 = sentence_split_head(list(record[1].values())[0][list_for_order[1][0]])
            head0_3, head1_3, sent3 = sentence_split_head(list(record[1].values())[0][list_for_order[2][0]])
            head0_4, head1_4, sent4 = sentence_split_head(list(record[1].values())[0][list_for_order[3][0]])
            
            dict_output[record[0]] = {"claim": search, "label": "NOT ENOUGH INFO", "evidence": [[head0_1, int(head1_1), sent1], [head0_2, int(head1_2), sent2], [head0_3, int(head1_3), sent3], [head0_4, int(head1_4), sent4]]}
            #dict_output[record[0]] = {"claim": search, "label": "NOT ENOUGH INFO", "evidence": [[head0_1, int(head1_1), sent1]]}
        except:
            dict_output[record[0]] = {"claim": search, "label": "NOT ENOUGH INFO", "evidence": []}
        #print(dict_output[record[0]])
        #print(sentence_split_head(list(record[1].values())[0][list_for_order[2][0]]))

        #print("\n\n")
with open('evidoneTop4.json', "wb") as of:
    of.write((json.dumps(dict_output, indent=4).encode("utf-8")))

