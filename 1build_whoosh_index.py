from whoosh.index import create_in
from whoosh.fields import *
from whoosh.index import open_dir
import nltk
import string
import os
import re
import json
import pickle
import time
print ("Start : %s" % time.ctime())
start = time.time()

schema = Schema(page_title=TEXT(stored=True),
                sentence_numbers=TEXT(stored=True))
                # Sentence_number is a string of numbers separated by whitespace

if not os.path.exists("index"):
    os.mkdir("index")
ix = create_in("index", schema)


# Function to read a wiki text file into lists
def read_wiki(file_path):
    page_titles = []

    sentence_numbers = []

    with open(file_path, 'r', encoding='utf-8') as f:

        for line in f:
            line_split = re.split(" ", line)
            page_titles.append(line_split[0].replace("_"," "))
            sentence_numbers.append((line_split[1]))

    return page_titles, sentence_numbers


def add_doc_toWhooshIndex(page_titles, sentence_numbers):
    ix = open_dir("index")
    writer = ix.writer()

    for i in range(len(page_titles)):
        page_title = page_titles[i]
        num = (sentence_numbers[i])
        # content = ' '.join(documents[i])
        writer.add_document(page_title=page_title,
                            sentence_numbers=num)

    print(" Complete adding documents, start write commit")
    writer.commit()


path = "./wikitext"

for dirpath, dirs, files in os.walk(path):

    for f in files:

        file_path = os.path.join(dirpath, f)
        print("file_path=", file_path)
        file_name = str(f)

        # Read wiki file
        page_titles, sentence_numbers= read_wiki(file_path)
        print(file_name + " read complete")

        # Add documents to Whoosh index
        add_doc_toWhooshIndex(page_titles, sentence_numbers)
        print(file_name + " write committed")







