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

            if dict_tfidf.get(word).get(record_num):
                dict_tfidf[word][record_num]["count"] += 1
            else:
                dict_tfidf[word].update({record_num : {"count" : 1 , "tfidf" : 0}})
        else:
            dict_tfidf[word] = {record_num : {"count" : 1 , "tfidf" : 0}}



def extract_text(xml_root, token, stop_words, ps, file):
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


def calc_query_bm25(the_query, inverted_index, docs_num):
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

    for doc in documents_vectors:
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

    inverted_index = corpus[dict_type]
    doc_reference = corpus["document_vector_len"]
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
        calc_query_bm25(the_query, inverted_index, docs_num)
    elif ranking == "tfidf":
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