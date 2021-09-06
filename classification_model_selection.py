import sys
import numpy as np
import pandas as pd
from  naive_bayes_algorithm import Naive_Bayes_Classifier
    
def model_scores(result):
    """
    calculate the model accuracy and error rate.
    function compares the actula class label versus predicted class label
    at each index. It sums all the true comparison results and divide by the 
    number of observations.
    @return model scores
    """  
    if result is None:
        raise ValueError( "The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")
       
   #save model model_scores
    metrics = {}
    metrics["accuracy"] = float(sum([p==a for p, a in zip(result["y_actual"], result["y_pred"])])/len(result))
   
    metrics["error_rate"] = 1-metrics["accuracy"]
   
    metrics["RMS"] = float(np.sqrt(sum([(p-a)**2 for p, a in zip(result["y_actual"], result["y_pred"])])/len(result)))
  
    tp = 0; tn = 0; fn = 0; fp = 0
    for y, y_hat in zip(result["y_actual"], result["y_pred"]):
        if y == 0 and y_hat == 0:
            tn += 1
        elif y == 0 and y_hat == 1:
            fp += 1
        elif y == 1 and y_hat == 1:
            tp += 1
        else:
            fn += 1
    return  metrics

class ModelSelctions():

    def stepwise_forward_selection(X_train, X_test, y_train, y_test, threshold = 0.0001):
        
        """
        first iterates through allthe features to find the feature
        that gives best model score metric. Keep the best found feature
        Perform pair-wise iteration of the best feature with all features to find the best two features
        iterate to find the best 3 features, so on. 
        In each iteration, check model score metric against previous score metric. 
        If the difference in core is not greater than the threshold, return all best features found.
        @return results data frame
        """
        if ((X_train is None) | (X_test is None) | (y_train is None) | (y_test is None)):
            raise ValueError( "The data sets must be assigned a non-nil reference to a Pandas DataFrame")
            
        #initialize model performance metric
        accuracy = 0
        accuracy_list = []

        error_rate = 0
        error_rate_list = []
        
        #save the best features into a list
        best_features_list = []
        
        feature_count = 0
        feature_count_list = []
        
        #save the number of features in each iteration step
        step_features_list = []

        while True:
            
            # forward step
            excluded = list(set(X_train.columns) - set(best_features_list))
    
            #intialize series to save performance metrics
            new_accuracy = pd.Series(index=excluded,dtype=float)
            new_error_rate = pd.Series(index=excluded,dtype=float)
                            
            for new_column in excluded:            
                sel_X_train = np.array(X_train[best_features_list + [new_column]]).astype(float)
                sel_X_test  = np.array(X_test[best_features_list + [new_column]]).astype(float)          
                        
                #fit the model
                model=Naive_Bayes_Classifier()
                model.fit(sel_X_train,y_train)
            
                #predict the data class
                model_pred=model.predict(sel_X_test)
            
                #add the actuals to our results data set
                model_pred["y_actual"] = y_test
            
                #calculate model accuracy
                current_perf = np.max(model_scores(model_pred)["accuracy"])
                
                #assign the current_perf to its feature
                new_accuracy[new_column] = current_perf
                #new_error_rate[new_column] = 1-new_accuracy[new_column]
            
            #find the best performance for the included + [new_column] round of iteration
            best_accuracy = new_accuracy.max()
            minimum_error_rate = 1- best_accuracy
            
            #calculate change in model performance
            perf_change = best_accuracy - accuracy
                   
            if perf_change > threshold:            
                accuracy     = best_accuracy
                error_rate = minimum_error_rate
                best_feature  = new_accuracy.idxmax()            
                feature_count = feature_count+1
                            
                best_features_list.append(best_feature)
                feature_count_list.append(feature_count)          
                step_features_list.append(str(best_features_list))
                accuracy_list.append(accuracy)
                error_rate_list.append(error_rate)

                print("features count =", feature_count)
                print("best feature =", best_feature)
                print("score with feature added =", accuracy)
                print("error with feature added =", error_rate)
            else:
                break
        
        results = pd.DataFrame()
        results["iter_features"] = step_features_list
        results["accuracy"]     = accuracy_list
        results["feature_count"] = feature_count_list
        results["best_features"] = best_features_list
        results["error_rate"] = error_rate_list
        
        return results