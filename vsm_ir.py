import os
import sys
import xml.etree.ElementTree as ET
import nltk
import math
import json
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

corpus = {}
dict_tfidf = {} # word : tf-idf score
dict_bm25 = {} # word : bm-25 score
dict_query = {}
doc_vec_len = {} # record id : len of document vector
words_num_in_file = {} # file_name : num of words in file after tokenization

### PART 1: Inverted Index and scores #################################################################


def update_doc_vec_len(record_num):
    if record_num not in doc_vec_len:
        doc_vec_len[record_num] = 0


def update_dictionary(text, record_num, file):
    for word in text:
        if word in dict_tfidf:
            if dict_tfidf.get(word).get(file):
                dict_tfidf[word][file]["count"] += 1
            else:
                dict_tfidf[word].update({file : {"count" : 1 , "tf_idf" : 0}})
        else:
            dict_tfidf[word] = {file : {"count" : 1 , "tf_idf" : 0}}



def extract_text(xml_root, token, stop_words, ps):
    for record in xml_root.findall("./RECORD"):
        text = ""
        text_without_stopwords = []
        for part in record:
            if part.tag == "RECORDNUM":
                record_num = int(part.text)
                update_doc_vec_len(record_num)
            if part.tag == "TITLE":
                text += str(part.text) + " "
            if part.tag == "ABSTRACT":
                text += str(part.text) + " "
            if part.tag == "EXTRACT":
                text += str(part.text) + " "

        text = text.lower()
        text = token.tokenize(text)  # tokenize and filter punctuation.
        text_without_stopwords = [word for word in text if not word in stop_words]  # remove stop words.

        for i in range(len(text_without_stopwords)):  # stemming
            text_without_stopwords[i] = ps.stem(text_without_stopwords[i])

        update_dictionary(text_without_stopwords, record_num)
        words_num_in_file[record_num] = len(text)


def extract_words(file):
    try:
        stop_words = set(stopwords.words("english"))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words("english"))

    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()
    xml_tree = ET.parse(file)
    root = xml_tree.getroot()

    extract_text(root, tokenizer, stop_words, ps)

# Build inverted index.
def calculate_tfidf(docs_num):
    for word in dict_tfidf:
        for file in dict_tfidf[word]:
            tf = dict_tfidf[word][file].get('count')/words_num_in_file.get(file)
            idf = math.log2(docs_num / len(dict_tfidf[word]))
            dict_tfidf[word][file]["tf_idf"] = tf*idf

            #Incrementing length of current file by (idf*tf)^2:
            doc_vec_len[file] += (tf*idf*tf*idf)


def calculate_bm25(docs_num):
    pass

def save_inverted_index_file():
    corpus["bm25_dict"] = dict_bm25
    corpus["tfidf_dict"] = dict_tfidf
    corpus["document_vector_len"] = doc_vec_len

    inverted_index_file = open("vsm_inverted_index.json", "w")
    json.dump(corpus, inverted_index_file, indent = 8)
    inverted_index_file.close()


def create_index(xmls_dir):
    input_dir = sys.argv[2]
    for file_name in os.listdir(xmls_dir):
        if file_name.endswith(".xml"):
            file=input_dir+"/"+file_name
            extract_words(file)
    docs_num = len(doc_vec_len)

    calculate_tfidf(docs_num)

    calculate_bm25(docs_num)

    for file in doc_vec_len:
        doc_vec_len[file] = math.sqrt(doc_vec_len[file])

    save_inverted_index_file()

#######################################################################################################
### PART 2: Retrieval information for given query #####################################################


# Return relavent docs based on given question.
def query():

   return None


if __name__ == '__main__':
    mood = sys.argv[1]
    if mood == "create_index":
        create_index(sys.argv[2])
    elif mood == "query":
        query()