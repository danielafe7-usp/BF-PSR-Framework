from multinomial_naive_bayes import MultinomialNaiveBayes
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
from nltk.corpus import stopwords
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import string
import csv
import re
from sklearn.metrics import f1_score
import numpy as np
import nltk 


stop = list(stopwords.words('english'))
porter = PorterStemmer()

def saving_pkl(file,name):

    with open(name,'wb') as f:
            pickle.dump(file,f)
            
def loading_pkl(name):
    with open(name, 'rb') as f:
        file = pickle.load(f)
    return file

def pre_processing_conversations(documents,labels):
    docs = []
    Y = []
    
    empty_docs = 0
    id = 0
    for conversation in (documents):
        tokens_clean_conversation = pre_processing(conversation)
        if len (tokens_clean_conversation) == 0:
            empty_docs += 1
        else:
            docs.append(tokens_clean_conversation)
            Y.append(labels[id])
        id += 1
    print("Empty conversations: ",empty_docs)
    return docs,Y

def stemSentence(sentence):
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def steamming_documents(documents):
    steaming_docs = list()
    for document in tqdm(documents):
        steam = stemSentence(document)
        #print(steam)
        #print("*"*20)
        steaming_docs.append(steam)
        
    return steaming_docs
        
def error_filtering(real_test_groomers,processed_labels,processed_predictions):
    # In this function we take all groomers removed by the preprocessing as Missclassified groomers.
    
    # Current groomers in test set: after filtering (0 loss, add manualy) and processing (x loss)
    current_groomers  = len(np.argwhere(processed_labels==1))
    #print("Current groomers in test: ",(current_groomers))
    

    filter_test_groomers = real_test_groomers - current_groomers
    
    print("Filter groomers: ",filter_test_groomers)
    #print("Including them into the confusion matrix as False Negatives ...")
    
    y = processed_labels
    new_predictions = processed_predictions
    # adding to true labels, groomers as 1's
    discard_groomers = np.ones(filter_test_groomers) 
    y = np.hstack((y,discard_groomers)) # We now add those discard groomers in the preprocessed labels
    false_negatives = np.zeros(filter_test_groomers)
    new_predictions = np.hstack((new_predictions,false_negatives)) # We add them as if we classify them wrontly (as 0's)
    assert(y.shape==new_predictions.shape)
    
    real_f1 = np.round(f1_score(y,new_predictions,average='binary',pos_label=1),4)
    return real_f1

def removing_sw(documents):
    stop = list(stopwords.words('english'))
    stop = set(stop)
    new_docs = list()
    for document in tqdm(documents):
        text_tokens = word_tokenize(document)
        tokens_without_sw = [word for word in text_tokens if not word in stop]
        tokens_without_sw = ' '.join(tokens_without_sw)
        new_docs.append(tokens_without_sw)
    return new_docs


def removing_empty(documents,labels):
    docs = []
    Y = []
    
    empty_docs = 0
    id = 0
    for conversation in (documents):
        conversation = conversation.lower()
        tokens = conversation.split()
        if len (tokens) == 0:
            empty_docs += 1
        else:
            docs.append(conversation)
            Y.append(labels[id])
        id += 1
        
    print("Emoty: ",empty_docs)
    return docs,Y

def n_grams(documents):
    documents_grams = []
    kval = 3
    for doc in documents:
        # get character ngrams        
        # Extract n-grams at the character level
        #'''
        cad = doc.replace(' ','&')
        ncad = ''
        ii = 0
        ij = kval
        tflag = 1
        while tflag:
            if ij+1 <= len(cad):
                ncad += ' ' +  cad[ii:ij]  + ' '
                ii = ii + 1
                ij = ij + 1
            else:
                tflag = 0
        cad = ncad
        documents_grams.append(cad)
    return documents_grams

def pre_processing(documents):
    # First pre-processing: Matlab
    documents = documents.replace('"','') 
    documents = documents.replace('.',' ') 
    documents = documents.replace("''",' ')
    documents = documents.lower()
    documents = documents.replace('[^\w\s]',' ') 
    
    # Second pre-processing: TMG

    # We convert from list to string
    i = documents
    # Remove punctuation as: !?&
    i = i.translate(str.maketrans('', '', string.punctuation))
    # Remove numerbs 
    i = re.sub(r'\b[0-9]+\b', '', i)#and alphanumerics too : re.sub('[0-9]+', '', i)
    # Remove words other than size 3
    words = [x for x in i.split() if len(x)>2 and len(x)<4]
    words = ' '.join(words)
    return words

def pre_processing_conversations(documents,labels):
    docs = []
    Y = []
    
    empty_docs = 0
    id = 0
    for conversation in (documents):
        tokens_clean_conversation = pre_processing(conversation)
        if len (tokens_clean_conversation) == 0:
            empty_docs += 1
        else:
            docs.append(tokens_clean_conversation)
            Y.append(labels[id])
        id += 1
        
    return docs,Y


def get_percentage_document(documents,retains):

    info_documents = []
    
    # Tokenize every single document and saving them in a matrix, we save the size of the doc to.
    start = time.time()
    for document in tqdm(documents):
        document_tokenize = nltk.word_tokenize(document)
        document_size = len(document_tokenize)
        info_documents.append([document_size,document_tokenize])
    end = time.time()
    print('Time: ',end-start)
    # Creating a set of documents, each of then with different percentages (%)
    final_documents = []
    for percent in retains:
        
        print("---------------------------------------- ",percent," ---------------------------------------------")
        reduce_documents = []
        if retains != 1:
            # Iteraits each doc and take only (%)
            for element in info_documents:
                document_size = element[0]
                document_tokenize = element[1]
                num_terms = round(document_size * percent)
                reduce_doc = document_tokenize[:num_terms]
                reduce_doc = ' '.join(reduce_doc)
                reduce_doc = reduce_doc.lower()
                reduce_documents.append(reduce_doc)
        else:
            print("100% of info") # In this case, we take everything
            for element in info_documents:
                document_size = element[0]
                document_tokenize = element[1]
                #num_terms = round(document_size * percent)
                reduce_doc = document_tokenize[:document_size] # In this case, we take everything
                reduce_doc = ' '.join(reduce_doc)
                reduce_doc = reduce_doc.lower()
                reduce_documents.append(reduce_doc)
        final_documents.append(reduce_documents)

    # final_documents= [[all_docs(1%)][all_docs(2%)][all_docs(3%)][...][all_docs(100%)]]
    return final_documents

def plotting(results,color,marker,name):
    #nbm = [58,60,61,62,63,64,65,66,67,68]
    time = np.arange(10) 
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xticks(np.arange(0, 10, 1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    #ax.plot(time, nbm, color="blue",linestyle='solid', linewidth=2.5, marker=marker,markersize = 12,markerfacecolor='#ececec',  label = "ORIGINAL")
    # BF-PSR
    for s in range((10)):
        results[s] = results[s] * 100
    lns_bfpsr_plus = ax.plot(time, results, color=color,linestyle='solid', linewidth=2.5, marker=marker,markersize = 12,markerfacecolor='#ececec',  label = name)
    lns = lns_bfpsr_plus

    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs)

    ax.set_xlabel("Percentage of information available")
    ax.set_ylabel(r"F1$_g$ measure")
    chunk_labels = ['10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']
    plt.title("State-of-the-art performance in the area of early grooming detection")
    ax.set_xticklabels(chunk_labels)
    plt.grid()
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
    fancybox=True, shadow=True,ncol=1,fontsize='medium')
    plt.show()
