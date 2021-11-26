from tqdm import tqdm
from collections import Counter
import numpy as np
import pandas as pd
import warnings
import pickle
from keras.preprocessing.text import Tokenizer
import csv
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn import metrics, preprocessing
import nltk
from nltk.tokenize import RegexpTokenizer
warnings.filterwarnings('ignore')
import time
from numpy import histogram
import scipy.cluster.vq as vq
from sklearn.cluster import KMeans
from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.feature_extraction.text import CountVectorizer
import gc
###################################################################

def saving_pkl(file,name):

    with open(name,'wb') as f:
            pickle.dump(file,f)
            
def loading_pkl(name):
    with open(name, 'rb') as f:
        file = pickle.load(f)
    return file

###################################################################

def plotting(results,color,marker,name):
    time = np.arange(10) 
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xticks(np.arange(0, 10, 1))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
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



###################################################################
