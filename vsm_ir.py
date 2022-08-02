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
# word : tf-idf score
dict_tfidf = {}
# scores of tf-idf query
dict_query = {}
# record id : the normalized len of document vector
doc_vec_len = {}
# file_name : num of words in file after tokenization
words_num_in_file = {}

### PART 1: Inverted Index and scores #################################################################


def update_doc_vec_len(record_num):
    """
    This function initializes the dict for corpus of documents and their vector length
    @:param record_num = the unique number for each document

    Result: @param doc_vec_len = initialize length vectors
    """

    if record_num not in doc_vec_len:
        doc_vec_len[record_num] = 0


def insert_to_dict(text, record_num, file):
    """
    This function insert to each word how many documents include it.
    Then initialize tf-idf of word and doc to zero
    @:param text = list of words after tokenization and stemming
    @:param record_num = the unique number for each document

    Result: @param dict_tfidf = holds updated data
    """

    for word in text:
        if word in dict_tfidf:

            if dict_tfidf.get(word).get(record_num):
                dict_tfidf[word][record_num]["count"] += 1
            else:
                dict_tfidf[word].update({record_num : {"count" : 1 , "tfidf" : 0}})
        else:
            dict_tfidf[word] = {record_num : {"count" : 1 , "tfidf" : 0}}



def extract_text(xml_root, token, stop_words, ps, file):
    """
    This function extracts text from the documents corpus and creates dict for each index
    @:param xml_root = xml root folder
    @:param token = tokenizer
    @:param stop_words = stock of stopwords
    @:param ps = PorterStemmer
    @:param file = folder from the file

    Result: updates @param text_without_stopwords and @param words_num_in_file
    """

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
        # tokenize and filter punctuation
        text = token.tokenize(text)
        # remove stopwords
        text_without_stopwords = [word for word in text if not word in stop_words]

        # stemming
        for i in range(len(text_without_stopwords)):
            text_without_stopwords[i] = ps.stem(text_without_stopwords[i])

        insert_to_dict(text_without_stopwords, record_num, file)
        words_num_in_file[record_num] = len(text)


def elicitation_data(file):
    """
    This main function of data elicitation from corpus

    Result: updates the dicts with the suitable data from corpus
    """

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


def calculate_tfidf(docs_num):
    """
    This function calculates the tf-idf for each word in the documents corpus
    This function creates the inverted index

    Result: @param dict_counter_occurrence = dict holds tf for each word in the query
    """

    for word in dict_tfidf:
        for file in dict_tfidf[word]:
            tf = dict_tfidf[word][file].get('count')/words_num_in_file.get(file)
            idf = math.log2(docs_num / len(dict_tfidf[word]))
            dict_tfidf[word][file]["tfidf"] = tf*idf

            #Incrementing length of current file by (idf*tf)^2:
            doc_vec_len[file] = doc_vec_len[file] + (tf*idf) ** 2

def save_inverted_index_file():
    """
    This function writes the data results for each document index as vector, documents length vector as well

    Result: creates inverted index file (json type)
    """

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
            elicitation_data(file)
    docs_num = len(doc_vec_len)

    calculate_tfidf(docs_num)

    for file in doc_vec_len:
        doc_vec_len[file] = math.sqrt(doc_vec_len[file])

    save_inverted_index_file()

#######################################################################################################
### PART 2: Retrieval information for given query #####################################################

def calc_number_of_docs_containing_term(inverted_index, the_query):
    """
    This function calculates the tf for each word in the query
    :return @param dict_counter_occurrence = dict holds tf for each word in the query
    """
    dict_counter_occurrence = {}
    counter = 0
    for word in the_query:
        if inverted_index.get(word) != None:
            for i in inverted_index[word]:
                counter += 1
            dict_counter_occurrence[word] = counter
            counter = 0
    return dict_counter_occurrence

def calc_idf_dict(query, docs_num, dict_counter_occurrence):
    """
    This function creates the dict that saves idf values for a given query from the user
    """
    dict = {}
    for word in query:
        if word in dict_counter_occurrence:
            # np.log(x) = ln(x)
            dict[word] = np.log((docs_num - dict_counter_occurrence[word] + 0.5) / (dict_counter_occurrence[word] + 0.5) + 1)

    return dict

def calc_numerator_bm25(idf, freq_word_doc, k):
    """
    This function calculates the numerator of the formula to calculate bm-25
    """

    return idf * freq_word_doc * (k+1)

def calc_denominator_bm25(freq_word_doc, k, b, doc_length, avgdl):
    """
    This function calculates the denominator of the formula to calculate bm-25
    """

    return freq_word_doc + k * (1-b + (b * (doc_length / avgdl)))

def calc_avgdl(dict_doc_lengths):
    """
    This function calculates the average length of the documents in the corpus
    """

    return np.array(list(dict_doc_lengths.values())).mean()

def sub_calc_bm25(freq_word_doc, k, b, doc_length, avgdl, idf):
    """"
    Sub main-function to calculate bm-25 for each word in the query
    @:param freq_word_doc = frequency of the word in the corpus
    @:param k = k1 was set for bm-25 calculation
    @:param b = b was set for bm-25 calculation
    @:param doc_length = number of documents in the corpus
    @:param avgdl = average document length in the corpus
    @:param idf = inverse document frequency of the word

    :return @param numerator / denominator = bm-25 for word in the query.
    """
    numerator = calc_numerator_bm25(idf, freq_word_doc, k)
    denominator = calc_denominator_bm25(freq_word_doc, k, b, doc_length, avgdl)
    return numerator / denominator


def calc_bm25(the_query, inverted_index, dict_doc_lengths, docs_num):
    """"
    Main function to calculate bm-25 ranking function
    @:param the_query = query from the user
    @:param inverted_index = inverted index from given XML documents
    @:param dict_doc_lengths = lengths of documents (saved in inverted index)
    @:param docs_num = number of documents in the corpus

    :return @param dict_for_bm25 = creates dictionary that saves for the given query,
    the numerical measurement for each document in the corpus according to bm-25
    """
    dict_for_bm25 = {}
    avgdl = calc_avgdl(dict_doc_lengths)
    dict_counter_occurrence = calc_number_of_docs_containing_term(inverted_index, the_query)
    dict_idf = calc_idf_dict(the_query, docs_num, dict_counter_occurrence)

    # k in range [1.2,2] - saw in lecture
    k = 2.432
    # b in range [0,1] - saw in lecture, usually b = 0.75
    b = 0.75

    for word in the_query:
        for i in range(1, 1240):
            if word in dict_counter_occurrence:
                idf = dict_idf[word]
            if inverted_index.get(word) != None:
                for j in inverted_index[word]:
                    if str(i) == j:
                        freq_word_doc = inverted_index[word][j]["count"]
                        bm25 = sub_calc_bm25(freq_word_doc, k, b, dict_doc_lengths[j], avgdl, idf)
                        if j in dict_for_bm25:
                            dict_for_bm25[j] = dict_for_bm25[j] + bm25
                        else:
                            dict_for_bm25[j] = bm25

    return dict_for_bm25

def calc_query_tfidf(the_query, inverted_index, docs_num):
    """"
    This function calculates the idf values and insert it to dict
    @:param the_query = query from the user
    @:param inverted_index = inverted index from given XML documents
    @:param docs_num = number of documents in the corpus

    result: updates dict_query according to calculated values
    """
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
            dict_query.update({str(i): tf*idf})


def calc_results(inverted_index, doc_reference, rank_type):
    """"
    This function calculates the final results according to tf-idf ranking function
    @:param the_query = query from the user
    @:param inverted_index = inverted index from given XML documents
    @:param doc_reference = document vector length

    :return @param results = creates dictionary that saves for the given query,
    the numerical measurement for each document in the corpus according to tf-idf
    """

    results = []

    # Calc query vector length
    query_len = 0
    for token in dict_query:
        query_len += (dict_query[token]*dict_query[token])
    query_len = math.sqrt(query_len)

    documents_vectors = {}

    for token in dict_query:

        w = dict_query[token]
        if inverted_index.get(token)!= None:
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
    """"
    This function responsible to operate process of given query from the user
    Then vector of the query calculated according to tokenization, stemming, tf-idf / bm-25 ranking values
    The function has 2 main sub-functions calc_query_bm25 and calc_query_tfidf that calculates grade according to given ranking function
    Each of them returns tuples of document and it's degree of similarity to the query
    Then writes the results to text file according to chosen best empirical thresholds
    @:param dict_type = user deciders how to calculate best matches according to bm-25 or tfidf ranking functions
    @:param index_path - index path file - source of corpus from user

    Result: writes to text file best documents (have biggest degree of similarity to the given query)
    """

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

    # Insert the json file to the global dictionary
    corpus = json.load(json_file)

    inverted_index = corpus["tfidf_dict"]
    doc_reference = corpus["document_vector_len"]
    docs_num = len(doc_reference)
    json_file.close()

    # clean query
    try:
        n = len(sys.argv)
        question = ""
        for i in range(4, n):
            question += sys.argv[i].lower()
            if i != n:
                question += " "
    except:
        print("query question is missing")
        return

    stop_words = set(stopwords.words("english"))
    tokenizer = RegexpTokenizer(r'\w+')
    ps = PorterStemmer()
    question = tokenizer.tokenize(question)
    the_query = [word for word in question if not word in stop_words]
    for i in range(len(the_query)):
        # stemming
        the_query[i] = ps.stem(the_query[i])

    # calculate scores for query
    if ranking == "bm25":
        dict_doc_lengths = corpus["bm25_dict"]
        ans_docs = calc_bm25(the_query, inverted_index, dict_doc_lengths, docs_num)
        f = open("ranked_query_docs.txt", "w")
        # sorting the dict of answer documents
        ans_docs = sorted(ans_docs.items(), key=lambda x: x[1], reverse=True)
        for i in ans_docs:
            if i[1] >= 6.65:
                f.write(i[0] + "\n")
        f.close()

    elif ranking == "tfidf":
        calc_query_tfidf(the_query, inverted_index, docs_num)
        # sorted before
        ans_docs = calc_results(inverted_index, doc_reference, ranking)
        f = open("ranked_query_docs.txt", "w")
        for i in range(0, len(ans_docs)):
            if ans_docs[i][1] >= 0.075:
                f.write(ans_docs[i][0] + "\n")
        f.close()


if __name__ == '__main__':

    mood = sys.argv[1]
    if mood == "create_index":
        create_index(sys.argv[2])
    elif mood == "query":
        query()