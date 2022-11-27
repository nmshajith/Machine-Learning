from __future__ import division
import numpy as np

class DecisionStump():
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.polarity = 1
        self.error = None
        self.alpha = None
    
    def predict(self, X_train):
        num_of_samples, num_of_features = X_train.shape
        y_pred = np.ones(num_of_samples)
        X_train_per_feature = X_train[:, self.feature]
        if self.polarity == 1:
            y_pred[ X_train_per_feature < self.threshold ] = -1
        else:
            y_pred[ X_train_per_feature >= self.threshold ] = -1
        return y_pred

class Adaboost():
    def __init__(self, num_of_stumps=5):
        self.num_of_stumps = num_of_stumps
        self.weights = None
        self.num_of_samples = 0
        self.num_of_features = 0
        self.X = None
        self.y = None
        self.stump_objs = [None for i in range(num_of_stumps)]

    def fit(self, X_train, y_train):
        self.num_of_samples, self.num_of_features = X_train.shape
        self.X = X_train
        self.y = y_train
        #Initialize weights with each sample's weight as 1/num_of_samples
        self.weights = np.full(self.num_of_samples, (1/self.num_of_samples))

        for _ in range(self.num_of_stumps):
            #For each decision stump, create a class obj
            stump = DecisionStump()
            #Get the best feature, threshold that can be choosen for that stump
            stump = self._get_best_feature_threshold(stump)
            #Calculate the Amount of say(alpha) for that stump
            stump.alpha = 0.5*np.log( (1-stump.error)/stump.error )
            #Predict X_train with that stump
            predictions = stump.predict(X_train)
            #Update the weights based on the predictions and amount of say of that stump
            self.weights *= np.exp(-stump.alpha*y_train*predictions)/ np.sum(self.weights)
            #Store that stump object in a list
            self.stump_objs[_] = stump

    def _get_best_feature_threshold(self, stump):
        min_error = np.Inf
        #For each feature and threshold in that feature
        for feature in range(self.num_of_features):
            X_per_feature = self.X[:,feature]
            thresholds = np.unique(X_per_feature)
            for threshold in thresholds:
                #Choose the y_pred and polarity with default values
                y_pred = np.ones(self.num_of_samples)
                polarity = 1
                #Based on the threshold, calculate the y_pred; 
                #i.e For whatever values<threshold; y_pred=-1; Else y_pred =1
                y_pred[ X_per_feature < threshold ] = -1
                #Calculate the error
                error = np.sum( self.weights[self.y!=y_pred], axis=0 )
                #If error>0.5, flip decision and error
                if error >0.5:
                    polarity = -1
                    error = 1-error
                #Find the best feature, threshold, by picking the one which has the min error
                if error<min_error:
                    min_error = error
                    stump.error = min_error
                    stump.polarity  = polarity
                    stump.threshold = threshold
                    stump.feature   = feature
        return stump
    
    def predict(self, X_test):
        #Predict the X_Test, with each of the stump created
        predictions = [stump.alpha*stump.predict(X_test) for stump in self.stump_objs]
        #Return the predictions based on the amount of say and predictions of each stump
        predictions = np.sign(np.sum(predictions, axis=0))
        return predictions


                