import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

import nltk
import math
import json
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import numpy as np

corpus = {}
dict_tfidf = {} # word : tf-idf score
dict_bm25 = {} # word : bm-25 score
dict_query = {}
doc_vec_len = {} # record id : len of document vector
words_num_in_file = {} # file_name : num of words in file after tokenization
tf_for_bm25 = {} # save tf for bm-25 score
avgdl = 0 # average document length in the corpus

### PART 1: Inverted Index and scores #################################################################


def update_doc_vec_len(record_num):
    if record_num not in doc_vec_len:
        doc_vec_len[record_num] = 0


def update_dictionary(text, record_num, file):
    for word in text:
        if word in dict_tfidf:

            if dict_tfidf.get(word).get(record_num):
                dict_tfidf[word][record_num]["count"] += 1
            else:
                dict_tfidf[word].update({record_num : {"count" : 1 , "tfidf" : 0}})
        else:
            dict_tfidf[word] = {record_num : {"count" : 1 , "tfidf" : 0}}



def extract_text(xml_root, token, stop_words, ps, file):
    global tf_for_bm25
    global avgdl
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

        update_dictionary(text_without_stopwords, record_num, file)
        words_num_in_file[record_num] = len(text)
        print(words_num_in_file)
        tf_for_bm25 = dict_tfidf.copy()
        #print(tf_for_bm25)
        #avgdl = np.array(list(words_num_in_file.values())).mean()


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

    extract_text(root, tokenizer, stop_words, ps, file)

# Build inverted index.
def calculate_tfidf(docs_num):
    for word in dict_tfidf:
        for file in dict_tfidf[word]:
            #print("word ",word, "file: ", file)
            #print(words_num_in_file)
            tf = dict_tfidf[word][file].get('count')/words_num_in_file.get(file)
            idf = math.log2(docs_num / len(dict_tfidf[word]))
            dict_tfidf[word][file]["tfidf"] = tf*idf

            #Incrementing length of current file by (idf*tf)^2:
            doc_vec_len[file] += (tf*idf*tf*idf)


def calculate_bm25(docs_num):
    pass

def save_inverted_index_file():
    #print(tf_for_bm25)
    corpus["bm25_dict"] = words_num_in_file
    corpus["tfidf_dict"] = dict_tfidf
    corpus["document_vector_len"] = doc_vec_len

    inverted_index_file = open("vsm_inverted_index.json", "w")
    json.dump(corpus, inverted_index_file, indent = 8)
    inverted_index_file.close()


def create_index(xmls_dir):
    input_dir = sys.argv[2]
    for file_name in os.listdir(xmls_dir):
        if file_name.endswith(".xml"):
            file = input_dir+"/"+file_name
            #print(file)
            extract_words(file)
    docs_num = len(doc_vec_len)
    #print(doc_vec_len)
    #print(docs_num)
    #print(dict_tfidf)

    calculate_tfidf(docs_num)

    calculate_bm25(docs_num)

    for file in doc_vec_len:
        doc_vec_len[file] = math.sqrt(doc_vec_len[file])

    #print((doc_vec_len))

    save_inverted_index_file()

#######################################################################################################
### PART 2: Retrieval information for given query #####################################################

def calc_number_of_docs_containing_term(inverted_index, the_query):
    dict_counter_occurrence = {}
    counter = 0
    for word in the_query:
        if inverted_index.get(word) != None:
            for i in inverted_index[word]:
                counter += 1
                #print(inverted_index[word][i]["count"])
            dict_counter_occurrence[word] = counter
            counter = 0
    return dict_counter_occurrence

def calculate_idf(word, docs_num, n_qi):
    return np.log((docs_num - n_qi + 0.5) / (n_qi + 0.5) + 1)

def calculate_idf_dict(query, docs_num, dict_counter_occurrence):
    dict = {}
    for word in query:
        dict[word] = np.log((docs_num - dict_counter_occurrence[word] + 0.5) / (dict_counter_occurrence[word] + 0.5) + 1)
    return dict

def calc_numerator_bm25(idf, freq_word_doc, k):
    return idf * freq_word_doc * (k+1)

def calc_denominator_bm25(freq_word_doc, k, b, doc_length, avgdl):
    return freq_word_doc + k * (1-b + b * (doc_length / avgdl))

def calculate_avgdl(dict_doc_lengths):
    return np.array(list(dict_doc_lengths.values())).mean()

def calc_bm25(freq_word_doc, k, b, doc_length, avgdl, idf):
    numerator = calc_numerator_bm25(idf, freq_word_doc, k)
    denominator = calc_denominator_bm25(freq_word_doc, k, b, doc_length, avgdl)
    return numerator / denominator


def calc_query_bm25(the_query, inverted_index, dict_doc_lengths, docs_num):
    sorted = {}
    dict_for_bm25 = {}
    results_for_bm25 = []
    avgdl = calculate_avgdl(dict_doc_lengths)
    dict_counter_occurrence = calc_number_of_docs_containing_term(inverted_index, the_query)
    dict_idf = calculate_idf_dict(the_query, docs_num, dict_counter_occurrence)

    # k in range [1.2,2] - saw in lecture, usually k = 1.2
    k = 1.2
    # b in range [0,1] - saw in lecture, usually b = 0.75
    b = 0.75
    query_length = len(the_query)
    sum = 0
    freq_word_doc = 0

    for i in range(1, 1240):
        #print(i)
        for word in the_query:
            #print(word)
            idf = dict_idf[word]
            #print(inverted_index.get(word))

            if inverted_index.get(word) != None:
                for j in inverted_index[word]:
                    #print(j)
                    if str(i) == j:
                        #print(i)
                        freq_word_doc = inverted_index[word][j]["count"]
                        bm25 = calc_bm25(freq_word_doc, k, b, dict_doc_lengths[j], avgdl, idf)
                        if j in dict_for_bm25:
                            dict_for_bm25[j] += bm25
                        else:
                            dict_for_bm25[j] = bm25
    #exit()
    #print(dict_for_bm25)
    #print(max(dict_for_bm25, key=dict_for_bm25.get))
    k = Counter(dict_for_bm25)
    high = k.most_common(10)
    for i in high:
        print(i[0], " :", i[1], " ")
    pass


def calc_query_tfidf(the_query, inverted_index, docs_num):
    query_length = len(the_query)
    for i in the_query:
        count = 0
        if dict_query.get(i) == None:
            for j in the_query:
                if i == j:
                    count += 1
            tf = (count / query_length)
            if inverted_index.get(i) != None:
                idf = math.log2(docs_num / len(inverted_index.get(i)))
            else:
                idf = 0
            dict_query.update({str(i) : tf*idf})


def calc_results(inverted_index, doc_reference, rank_type):
    results = []

    # Calc query vector length
    query_len = 0
    for token in dict_query:
        query_len += (dict_query[token]*dict_query[token])
    query_len = math.sqrt(query_len)

    documents_vectors = {}

    for token in dict_query:

        w = dict_query[token]
        if inverted_index.get(token) != None:
            for doc in inverted_index[token]:
                if doc not in documents_vectors:
                    documents_vectors[doc] = 0

                documents_vectors[doc] += (inverted_index[token][doc][rank_type] * w)

    #print(documents_vectors)
    for doc in documents_vectors:
        #print(doc)
        doc_query_product = documents_vectors[doc]
        doc_len = doc_reference[doc]
        cosSim = doc_query_product / (doc_len * query_len)
        results.append((doc, cosSim))

    # Sort list by cos similarity
    results.sort(key = lambda x: x[1], reverse=1)
    return results


def query():
    dict_type = ""
    ranking = sys.argv[2]
    if ranking == "bm25":
        dict_type = "bm25_dict"
    elif ranking == "tfidf":
        dict_type = "tfidf_dict"
    else:
        print("wrong ranking type from user")
        return

    index_path = sys.argv[3]

    try:
        json_file = open(index_path,"r")
    except:
        print("wrong index path from user")
        return

    corpus = json.load(json_file) # Insert the json file to the global dictionary.

    inverted_index = corpus["tfidf_dict"]
    #print(inverted_index)
    doc_reference = corpus["document_vector_len"]
    #print(doc_reference)
    docs_num = len(doc_reference)
    json_file.close()

    #clean query
    try:
        #print(len(sys.argv))
        n = len(sys.argv)
        question = ""
        for i in range(4, n):
            question += sys.argv[i].lower()
            if i != n:
                question += " "
        #print(question)
    except:
        print("query question is missing")
        return

    stop_words = set(stopwords.words("english"))
    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()
    question = tokenizer.tokenize(question)
    the_query = [word for word in question if not word in stop_words]
    for i in range(len(the_query)): #stemming
        the_query[i] = ps.stem(the_query[i])

    # calculate scores for query
    if ranking == "bm25":
        dict_doc_lengths = corpus["bm25_dict"]
        calc_query_bm25(the_query, inverted_index, dict_doc_lengths, docs_num)
    elif ranking == "tfidf":
        #print(inverted_index)
        calc_query_tfidf(the_query, inverted_index, docs_num)

    ans_docs = calc_results(inverted_index, doc_reference, ranking)

    f = open("ranked_query_docs.txt","w")
    for i in range(0, len(ans_docs)):
        if(ans_docs[i][1] >= 0.075):
            f.write(ans_docs[i][0] + "\n")

    f.close()

if __name__ == '__main__':

    mood = sys.argv[1]
    if mood == "create_index":
        create_index(sys.argv[2])
    elif mood == "query":
        query()