import numpy as np
import pandas as pd

class Naive_Bayes_Classifier():
    def __init__(self):
        
        #save the classes and their data
        self.data_class={}    
        
    def fit(self,X_train,y_train):
        
        def group_data_to_classes(data_class,X_train,y_train):
            class0=True
            class1=True
            
            for i in range(y_train.shape[0]):
                X_temp=X_train[i,:].reshape(X_train[i,:].shape[0],1)
                if y_train[i]==0:
                    if class0==True:
                        data_class[0]=X_temp
                        class0=False
                    else:
                        data_class[0]=np.append(data_class[0],X_temp,axis=1)
                elif y_train[i]==1:
                    if class1==True:
                        data_class[1]=X_temp
                        class1=False
                    else:
                        data_class[1]=np.append(data_class[1],X_temp,axis=1)
            
            return data_class
        
        #set the train set and target 
        self.X_train=X_train
        self.y_train=y_train
        
        #initialize data array
        self.data_class[0]=np.array([[]])
        self.data_class[1]=np.array([[]]) 
        
        #find data and their classess
        self.data_class=group_data_to_classes(self.data_class,self.X_train,self.y_train)
        
        self.data_class[0]=self.data_class[0].T
        self.data_class[1]=self.data_class[1].T

        #calculate the means for the train set
        self.mean_1=np.mean(self.data_class[0],axis=0)
        self.mean_2=np.mean(self.data_class[1],axis=0)
        
        #calculate the standard deviation for the train set
        self.std_1=np.std(self.data_class[0],axis=0)
        self.std_2=np.std(self.data_class[1],axis=0)

    def predict(self, X_test):
        """
        For numerical data modeled as a normal distribution,
        we can use the Gaussian/normal distribution function to calculate likelihood
        """
        
        def calc_posterior(X, X_train_class, mean_, std_):
            
            def class_likelihood(x, mean, std): 
                #use the normal pdf to calculate the likelihood
                lieklihood = (np.sqrt(2*np.pi*std)**-1)*np.exp(-(x-mean)**2/(2*std**2))
                return lieklihood
            
            #product of class likelihoods for all features in the data
            likelihood_prod = np.prod(class_likelihood(X,mean_,std_),axis=1)
            
            #class prior
            prior = X_train_class.shape[0]/self.X_train.shape[0]

            #class posterior distribution
            posterior=likelihood_prod*prior
            
            return posterior
        
        #class 0 posterior
        class_0=calc_posterior(X_test,self.data_class[0],self.mean_1,self.std_1)
        
        #class 1 posterior
        class_1=calc_posterior(X_test,self.data_class[1],self.mean_2,self.std_2)
        
        
        #find the class that each data row belongs to
        y_pred =[]
        for i, j in zip(class_0, class_1):
            if (i > j):
                y_pred.append(0)
            else:
                y_pred.append(1) 
                
        #store data to a dataframe to return        
        results = pd.DataFrame()        
        results["class_0_posterior"] = class_0
        results["class_1_posterior"] = class_1
        results["y_pred"] = y_pred
        
        return results


