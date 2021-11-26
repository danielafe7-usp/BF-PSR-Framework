 
from sentiment_reader import SentimentCorpus
from multinomial_naive_bayes import MultinomialNaiveBayes
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    dataset = SentimentCorpus()
    nb = MultinomialNaiveBayes()
    
    """
    params = nb.train(dataset.train_X, dataset.train_y)
    #loglikelihood of each word per class  row= no of words columns =classes
    predict_train = nb.test(dataset.train_X, params)
    #predict_train has the output of test doc belonging to which class
    eval_train = nb.evaluate(predict_train, dataset.train_y)
    
    predict_test = nb.test(dataset.test_X, params)
    eval_test = nb.evaluate(predict_test, dataset.test_y)
    
    print ("Accuracy on training set: %f, on test set: %f" % (eval_train, eval_test))
    """

    corpus_train = [
    'Chinese Beijin Chinese',
    'Chinese Chinese Shangai',
    'Chinese Macao',
    'Tokio Japan Chinese',
    ]
    labels_train = [0,0,0,1]
    corpus_test = [
    'Chinese Chinese Chinese Tokio Japan',
    'Chinese Chinese Chinese Tokio Shangai',
    'This is mua'
    ]
    labels_test = [0,0,1]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus_train)
    Dicc = vectorizer.get_feature_names()
    print(vectorizer.get_feature_names())
    X_train = X.toarray()
    print("X_train")
    print(X_train)
    print("*"*40)
    del vectorizer
    vectorizer = CountVectorizer(vocabulary=Dicc)
    X = vectorizer.fit_transform(corpus_test)
    print("X_test")
    X_test = X.toarray()
    print(X_test)

    params = nb.train(X_train, labels_train)
    #loglikelihood of each word per class  row= no of words columns =classes
    ##predict_train = nb.test(X_train, params)
    #predict_train has the output of test doc belonging to which class
    ##eval_train = nb.evaluate(predict_train, labels_train)
    
    predict_test = nb.test(X_test, params)
    eval_test = nb.evaluate(predict_test, labels_test)
    
    #print ("Accuracy on training set: %f, on test set: %f" % (eval_train, eval_test))
    print ("Accuracy on test set: %f" % (eval_test))
    


