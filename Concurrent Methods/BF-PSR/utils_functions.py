import numpy as np
import pandas as pd
import pickle
import warnings
from nltk.corpus import stopwords 
import re
from somajo import SoMaJo
from collections import defaultdict
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
import matplotlib.ticker as mtick

warnings.filterwarnings('ignore')

#############################################

def saving_pkl(file,name):

    with open(name,'wb') as f:
            pickle.dump(file,f)
            
def loading_pkl(name):
    with open(name, 'rb') as f:
        file = pickle.load(f)
    return file

#############################################   
#en_PTB : english language
tokenizer = SoMaJo(language="en_PTB",split_camel_case=False,split_sentences=False)
def tokenizer_Somajo_vectorizer(document):
            
    sentences = tokenizer.tokenize_text([document])
    token_l = []
    for sentence in sentences:
        for token in sentence:
            
            '''Allows symbols'''
            if token.token_class == "XML_entity" or token.token_class == "XML_tag":
                continue # We ignore this tokens because are noise
  
            # Lowercase the tokens except the emoticons
            if token.token_class != "emoticon":
                token = (token.text).lower()
            else: 
                token = (token.text) 
            
            token_l.append(token)
    
    return token_l

#############################################  

def pre_processing_conversations(nc,documents,labels,time_bf,participants_bf,inte_user,keep_empty= False):
    docs = []
    Y = []
    time = []
    n_participants = []
    n_int_user = []
    
    empty_docs = 0
    index = 0
    indexs_l = []
    for id,conversation in (enumerate(documents)):
        tokens_clean_conversation = nc(conversation)
        if len (tokens_clean_conversation) == 0:
            empty_docs += 1
            indexs_l.append(index)
            
        else:
            docs.append(tokens_clean_conversation)
            Y.append(labels[id])
            time.append(time_bf[id])
            n_participants.append(participants_bf[id])
            n_int_user.append(inte_user[id])
        index += 1
    if keep_empty == False:
        return docs,Y,time,n_participants,n_int_user
    else:
        return indexs_l

#############################################  

def pre_processing_conversations_keep_empty_docs(nc,documents,labels,time_bf,participants_bf,inte_user):
    docs = []
    Y = []
    time = []
    n_participants = []
    n_int_user = []
    
    empty_docs = 0
    for id,conversation in (enumerate(documents)):
        tokens_clean_conversation = nc(conversation)
        if len (tokens_clean_conversation) == 0:
            empty_docs += 1
            docs.append(None)
            Y.append(None)
            time.append(None)
            n_participants.append(None)
            n_int_user.append(None)
        else:
            docs.append(tokens_clean_conversation)
            Y.append(labels[id])
            time.append(time_bf[id])
            n_participants.append(participants_bf[id])
            n_int_user.append(inte_user[id])
    return docs,Y,time,n_participants,n_int_user

############################################# 

def calculating_vocabulary(doc_train):
    # Fundamental modification to the PSR method
    min_df = 3 

    vocabulary_dict = defaultdict(int)
    for tokenize_document in doc_train:
        for token in tokenize_document:
            vocabulary_dict[token] = vocabulary_dict.get(token,0) + 1
    vocabulary = []
    for key,values in vocabulary_dict.items():
        if values >= min_df:
            vocabulary.append(key)
    vocabulary = sorted(vocabulary)
    return vocabulary
    
#############################################  

def calculated_df(documents,vocabulary):
    DF = {}
    N = len(documents)
    for i in (range(N)):
        tokens = (list(set.intersection(set(vocabulary),set(documents[i])))) 
        for w in tokens:
            try:
                DF[w].add(i)
            except:
                DF[w] = {i}

    for i in DF:
        DF[i] = len(DF[i])
    return DF

#############################################  
def calculating_vocabulary(doc_train):
    # Fundamental modification to the PSR method
    min_df = 3 

    vocabulary_dict = defaultdict(int)
    for tokenize_document in doc_train:
        for token in tokenize_document:
            vocabulary_dict[token] = vocabulary_dict.get(token,0) + 1
    vocabulary = []
    for key,values in vocabulary_dict.items():
        if values >= min_df:
            vocabulary.append(key)
    vocabulary = sorted(vocabulary)
    return vocabulary

#############################################  


def PSR_weight_schema(documents,labels,vocabulary,DF):
    '''
    Args:
        documents : conversations tokenized
        labels : groomer and  non-groomer
        vocabulary : list of unique tokens
    Outputs:
        PSR --> words_weights : dic[token] = [NG,G]
    '''
    number_classes = 2

    # Initialize the w_tokens dictionary
    words_weights = dict() 
    for v in vocabulary:
        words_weights[v] =  [0,0]

    N = len(documents)
    
    # Calculating frequency (TF) for each token for each class
    for label,document in zip(labels,documents):
        
        # Group each unique token and its frequency Exemp: counter[hi] = 45
        counter = Counter(document)

        for token in np.unique(document):
            if token not in vocabulary:
                continue 
                
            
            '''
            Weighting schema tfidf
            
            '''
            tf = counter[token]
            tf = 1 + np.log(tf) # Sublinear tf
            df = DF.get(token,0)
            idf = np.log((N)/df) + 1  
            tf_idf = tf*idf
            w_token = np.log2((1+ (tf_idf / len(document))))
            words_weights[token][label] += w_token # Only give weights to tokens that belong to the vocabulary (obtain with train set)
            
            
    # First normalization over words_weights ; it avoids unbalance data (sum by colum)
    df = pd.DataFrame.from_dict(words_weights,orient = 'index')
    X = df.values
    X_std = X / (X.sum(axis=0,keepdims=True)) 
    # Second normalization ; make its comparable betwenn class p << m (sum by row)
    X_std = X_std / (X_std.sum(axis=1,keepdims=True)) 
    id = 0
    for key,values in words_weights.items():
        words_weights[key] = np.nan_to_num(X_std[id])
        id += 1
    return words_weights


#############################################  

def PSR_plus_document_representation(documents,psr_weights):
    '''
    Args:
        documents : tokenized documents
        labels : groomer non-groomer
        psr_weights : dictionary with list of tokens and their PSR weights
    Outputs:
        DR/DRT : documents in Profile representation (vector of size 2)
    '''
    number_classes = 2
    DR = np.zeros((len(documents),number_classes,))
    id = 0
    for tokenize_document in (documents):
        for token in tokenize_document:
            if token not in psr_weights:
                continue
            # Document representatio is equal to the sum of all tokens (t_i)
            # Note summ every time == TF * t_i
            DR[id] += psr_weights[token]
        # Normalization 
        DR[id] = (DR[id] / len(tokenize_document))
        # Next document
        id += 1
    return DR

############################################# 

def plotting(document_representation,labels,flag="g_ng"):
    groomers = []
    non_groomers = []
    ng_ng,g_ng,ng_g,g_g = [],[],[],[]

    # Obtaining Groomer and Non-groomers values
    id = 0
    for i in labels:
        if i==1:
            groomers.append(document_representation[id])
        else:
            non_groomers.append(document_representation[id])
        id += 1

    for ng,g in non_groomers:
        ng_ng.append(ng)
        g_ng.append(g)
        #print(ng," ",g)

    for ng,g in groomers:
        ng_g.append(ng)
        g_g.append(g)
        #print(ng," ",g)
        
    print(len(groomers)," ",len(non_groomers))   

    # Plot the samples using columns 1 and 2 of the matrix
    fig, ax = plt.subplots(figsize = (8, 8))

    colors = ['#297fb8','#e74d3d']

    # Color base on sentiment
    if flag == "g_ng":
        ax.scatter(ng_ng,g_ng, color="blue", s = 0.1,marker = "*")  # Plot a dot for each pair of words
        ax.scatter(ng_g,g_g, color="red", s = 0.1,marker = "D")  # Plot a dot for each pair of words

    if flag == "g":
        
        ax.scatter(ng_g,g_g, color="red", s = 0.1,marker = "D")  # Plot a dot for each pair of words

    if flag == "ng":
        ax.scatter(ng_ng,g_ng, color="blue", s = 0.1,marker = "*")  # Plot a dot for each pair of words

    plt.xlabel("Positive") # x-axis label
    plt.ylabel("Negative") # y-axis label

    ax.legend()
    plt.show()
    
############################################# 

def calculating_time(feature,n_features):
    time_feature = np.zeros((len(feature),n_features)) # Number of conversations x num features
    id = 0
    for i in feature:
        hours,minutes  = i.split(":")
        time_feature[id][0],time_feature[id][1] = int(hours), int (minutes)
        id +=1
    # Normalizing data
    normalize_time_feature = my_min_max(time_feature)
    return normalize_time_feature

############################################# 

def my_min_max(matrix):
    max_value = matrix.max()
    min_value = matrix.min()
    normalize_matrix = (matrix - min_value) / (max_value - min_value)    
    return normalize_matrix

############################################# 

def calculating_CSW(tokenize_documents):
    set_words = set((wn.words()))
    ww_bf = np.zeros((len(tokenize_documents),1))
    
    id = 0
    for document in (tokenize_documents):
        wornet = 0
        counter = Counter(document)
        
        for k,v in counter.items():
            if k in set_words:
                wornet += v
            
        ww_bf[id] = wornet / len(document)
        id += 1
   
    return ww_bf

############################################# 

def calculating_sexual_words(words, tokenize_documents):
    set_words = set((wn.words())) 
    ww_bf = np.zeros((len(tokenize_documents),1))
    id = 0
    for document in (tokenize_documents):#tqdm
        tw = 0
        wordnet = 0
        counter = Counter(document)
        
        for k,v in counter.items():
            if k in set_words:
                wordnet += v
                
        for token in words:
            tw += counter.get(token,0)
        ww_bf[id] = tw / len(document) 
        id += 1
    return ww_bf

############################################# 

def calculating_emotional_markers(words, tokenize_documents,n_emo):
    ww_bf = np.zeros((len(tokenize_documents),n_emo))
    id = 0
    for document in (tokenize_documents):
        tw = np.zeros((1,n_emo))
        total = 0
        counter = Counter(document)
        
        for k,v in counter.items():
            tw += words.get(k,0) * v 
            if k in words:
                total += 1
        ww_bf[id] = tw / len(document) 
        id += 1
    return ww_bf

############################################# 
def processing_depeche_lexicon(depeche_emotions):
    tmp_dict = dict()
    for key in depeche_emotions.keys():
        max_index = depeche_emotions[key].argmax()
        tmp = np.zeros(8) # Depeche lexicon has 8 emotions
        tmp[max_index] = 1
        tmp_dict[key] = tmp
    return tmp_dict

############################################# 

def calculating_emoticons_set(documents_tokenize,emoticons_list):
    documents = [" ".join(document) for document in documents_tokenize ]
    tokenizer = SoMaJo(language="en_PTB",split_camel_case=False,split_sentences=False)
    sentences = tokenizer.tokenize_text(documents)
    
    emoticons_matrix = np.zeros((len(documents),1))
    id = 0
    for sentence in (sentences):
        emoticon,total = 0,0
        counter = Counter(sentence)
        for token,v in counter.items():
            if  token.token_class == "emoticon":
                emoticon += v
                emoticons_list.append(token.text)
            
        total = len(sentence)
        emoticons_matrix[id] = emoticon / total
        id += 1
        
    return emoticons_matrix,emoticons_list

############################################# 

def calculating_emoticons_faster(emoticons_set, tokenize_documents):
    emoticons_matrix = np.zeros((len(tokenize_documents),1))
    id = 0
    for document in (tokenize_documents):
        emoticons,total = 0,0
        counter = Counter(document)
        for token,v in counter.items():
            if  token in emoticons_set:
                emoticons += v

        total = len(document)
        emoticons_matrix[id] = emoticons / total
        id += 1
    return emoticons_matrix

############################################# 

def blackbox_classifier(training_set,testing_set,training_labels,testing_labels):
    clf = MLPClassifier(random_state=42, max_iter=600) 
    clf_build = clf.fit(training_set, training_labels)
    predictions = clf_build.predict(testing_set)
    
    # Performing F1 mesure of the positive (Groomer) class
    f1_g = np.round(f1_score(testing_labels,predictions,average='binary',pos_label=1),4)
    real_f1_g = error_filtering(testing_labels,predictions)

    return real_f1_g, clf_build

############################################# 

def perform_predictions(trained_model,partial_information,true_labels_partial):
    predictions = trained_model.predict(partial_information) 
    # Performing F1 mesure of the positive (Groomer) class
    #f1_g = np.round(f1_score(true_labels_partial,predictions,average='binary',pos_label=1),4)
    real_f1_g = error_filtering(true_labels_partial,predictions)
    return real_f1_g

############################################# 

def BF_PSR(DR,DRT,Y_train,Y_test,BFS,labels_BF,num_BF):
    
    '''
        Input:
            DR: Profiles (groomer/non-groomer) obtained with the PSR++ method and the training set
            DRT: Profiles (groomer/non-groomer) obtained with the PSR++ method and the testing set
            Y_train/Y_test : Labels of the conversations
            BFS : The proposed behavioral features 
            labels_BF : List of the proposed BF
            num_BF : Number of the proposed BF
        Output:
            F1_g : Return de F1 mesure of the positive (groomer) class
            trained_model : Trained model to perform predictions in a future
    '''
    
    # Initialize the BF_PSR vector with the profiles
    BF_PSR_train = DR
    BF_PSR_test = DRT
    # Horizontal stack of the proposed BFs
    for i in range(num_BF):
        bf_train = BFS[i][0]
        bf_test = BFS[i][1]
        print(i,") Stacking ",labels_BF[i]," BF of shape: ",bf_train.shape)
        #print(bf_train.shape," ",bf_test.shape)
        BF_PSR_train = np.hstack((BF_PSR_train,bf_train))
        BF_PSR_test = np.hstack((BF_PSR_test,bf_test))
    print("The final PF-PSR vector is of shape: ",BF_PSR_train.shape," ",BF_PSR_test.shape)
    
    f1_g,trained_model = blackbox_classifier(BF_PSR_train,BF_PSR_test,Y_train,Y_test)
    return f1_g, trained_model
    
############################################# 

def BF_PSR_predictions(model,DRT_partial,Y_partial,BFS,labels_BF,num_BF):
    
    '''
        Input:
            model : Train classfier with 100% of the available information
            DRT_partial: Profiles (groomer/non-groomer) obtained with the PSR++ method and the testing set
            Y_partial : Labels of the conversations
            BFS : The proposed behavioral features
            labels_BF : List of the proposed BF
            num_BF : Number of the proposed BF
        Output:
            F1_g : Return de F1 mesure of the positive (groomer) class
    '''
    
    # Initialize the BF_PSR vector with the profiles
    BF_PSR_partial = DRT_partial
    # Horizontal stack of the proposed BFs
    for i in range(num_BF):
        bf_partial = BFS[i]
        print(i,") Stacking ",labels_BF[i]," BF")
        BF_PSR_partial = np.hstack((BF_PSR_partial,bf_partial))
    print("The final BF-PSR vector is of size: ",BF_PSR_partial.shape)
    
    f1_g = perform_predictions(model,BF_PSR_partial,Y_partial)
    return f1_g
    
############################################# 

def error_filtering(processed_labels,processed_predictions):
    # In this function we take all groomers removed by the preprocessing as Missclassified groomers.
    
    # Current groomers in test set: after filtering (0 loss, add manualy) and processing (x loss)
    current_groomers  = len(np.argwhere(processed_labels==1))
    #print("Current groomers in test: ",(current_groomers))
    
    # Remember that state-of-art methods do not take into account filtered groomers
    real_test_groomers = 3724 # Fix number, before filtering and processing ( 3737 not because it has empty documents)
    filter_test_groomers = real_test_groomers - current_groomers
    
    #print("Filter groomers: ",filter_test_groomers)
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

############################################# 


def obtain_text(colum):
    '''
    Args: colum is representaing the percentage (10%,20%,100%)
    '''
    first_ = np.concatenate(colum)
    print(first_.shape)
    return first_

############################################# 

def pre_processing_chunks(nc,retains,Ys_test,Xss_test_time,keep_empty=False):
    interaction_words_bf = loading_pkl('dataBase/test_early_interaction_words_user.pkl') 
    number_participants_bf = loading_pkl('dataBase/participants_matrix.pkl')#loading_pkl('dataBase/test_early_number_participants.pkl') 
    text_percentages_text = loading_pkl('dataBase/test_early_text.pkl')
    percentage = 0
    for retain in (retains):
        # Each chunk contains  the x% of every conversation, interaction words per user BF and number of participants BF  (until that point of the chat)
        print("---------------------- ",retain," ----------------------")
        text = text_percentages_text[:,[percentage]]
        Xss_test = obtain_text(text)
        Xss_test_participants = number_participants_bf[:,[percentage]]
        Xss_test_int_user = interaction_words_bf[:,[percentage]]
                
        
        print(Xss_test.shape," ",Xss_test_participants.shape," ",Xss_test_int_user.shape)

        # Processing the set of documents
        if keep_empty == False:
            doc_partial,Y_partial,X_test_time_partial,X_test_participants_partial,X_test_int_user_partial = pre_processing_conversations(nc,Xss_test,Ys_test,Xss_test_time,Xss_test_participants,Xss_test_int_user)
            print("Test Documents, There are :",len(doc_partial)," ",len(Y_partial))
            Y_partial = np.asarray(Y_partial)
            print('\033[94m')
            chunk_info = [doc_partial,Y_partial,X_test_time_partial,X_test_participants_partial,X_test_int_user_partial]
            saving_pkl(chunk_info,"Pre_pro_chunks/chunk_preprocessed_"+str(percentage))
        
        else:
            doc_partial,Y_partial,X_test_time_partial,X_test_participants_partial,X_test_int_user_partial = pre_processing_conversations_keep_empty_docs(nc,Xss_test,Ys_test,Xss_test_time,Xss_test_participants,Xss_test_int_user)
            print("Test Documents, There are :",len(doc_partial)," ",len(Y_partial))
            Y_partial = np.asarray(Y_partial)
            print('\033[94m')

            chunk_info = [doc_partial,Y_partial,X_test_time_partial,X_test_participants_partial,X_test_int_user_partial]
            saving_pkl(chunk_info,"Keep_empty_chunks/empty_chunk_preprocessed_"+str(percentage))
        
        percentage += 1
        
############################################# 

def column_extract(column):
    column_correct = np.zeros((len(column),5))
    id = 0
    for element in column:
        column_correct[id] = element[0]
        id += 1
    return column_correct

############################################# 

def plotting_bf_psr(bf_psr):
    time = np.arange(10) 
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xticks(np.arange(0, 10, 1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # BF-PSR
    for s in range((10)):
        bf_psr[s] = bf_psr[s] * 100
    lns_bfpsr_plus = ax.plot(time, bf_psr, color="black",linestyle='solid', linewidth=2.5, marker='H',markersize = 12,markerfacecolor='#ececec',  label = "BF-PSR (our proposal)")
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

############################################# 

'''Adding new functions for the PJZ and PJZC datasets ...'''

def error_filtering_PJdataset(processed_labels,processed_predictions):
    print("Calculating error ...")
    # In this function we take all groomers removed by the preprocessing as Missclassified groomers.
    
    # Current groomers in test set: after filtering (0 loss, add manualy) and processing (x loss)
    current_groomers  = len(np.argwhere(processed_labels==1))
    print("Current groomers in test: ",(current_groomers))
    
    # Remember that state-of-art methods do not take into account filtered groomers
    real_test_groomers = 1104 # Fix number, in PJZ and PJZC are 1104 groomers :)
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


def pre_processing_conversations_new_datasets(nc,documents,time_bf,participants_bf,inte_user,labels=[],_predict=False):
    docs = []
    time = []
    n_participants = []
    n_int_user = []
    n_labels = []
    
    empty_docs = 0
    for id in range(len(documents)):
        tokens_clean_conversation = nc(documents[id])
        if len (tokens_clean_conversation) == 0:
            empty_docs += 1
        else:
            docs.append(tokens_clean_conversation)
            time.append(time_bf[id])
            n_participants.append(participants_bf[id])
            n_int_user.append(inte_user[id])
            if _predict:
                n_labels.append(labels[id])
            
            
    if _predict:
        return docs,time,n_participants,n_int_user,n_labels
    else:
        return docs,time,n_participants,n_int_user
    