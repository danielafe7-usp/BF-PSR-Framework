import numpy as np


class LinearClassifier():

    def __init__(self):
        self.trained = False

    def train(self,x,y):
        '''
        Returns the weight vector
        '''
        raise NotImplementedError('LinearClassifier.train not implemented')

    def get_scores(self,x,w):
        '''
        Computes the dot product between X,w
        '''
        return np.dot(x,w)

    def get_label(self,x,w):
        '''
        Computes the label for each data point
        '''
        #scores = np.dot(x,w)
        scores_l = list()
        for s in range(len(x)):
            score = np.power(w.T,x[s])
            score = np.prod(score,axis=1)
            score_label = np.argmax(score)
            #score = np.expand_dims(score, axis=0)
            scores_l.append(score_label)
        
        #print(scores_l)
        return scores_l
        #return np.argmax(scores,axis=1).transpose() #return the class that each documents has a higher probabilty to be in

    def test(self,x,w):
        '''
        Classifies the points based on a weight vector.
        '''
        #x - train_x
        if self.trained == False:
            raise ValueError("Model not trained. Cannot test")
            return 0
        x = self.add_intercept_term(x)
        return self.get_label(x,w)
    
    def add_intercept_term(self,x):
        ''' Adds a column of ones to estimate the intercept term for separation boundary'''
        nr_x, nr_f = x.shape
        #adds 1 column to make dot product(i.e matrix multiplication) between train_x and params.
        #as params has 1 extra row of prior
        intercept = np.ones([nr_x,1])
        x = np.hstack((intercept,x))# adding extra column on left of train_x
        return x

    def evaluate(self,truth,predicted):
        correct = 0.0
        total = 0.0
        for i in range(len(truth)):
            if(truth[i] == predicted[i]):
                correct += 1
            total += 1
        return 1.0*correct/total
