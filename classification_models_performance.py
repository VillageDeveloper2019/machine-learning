import numpy as np
import pandas as pd

class Model_Performance():
    
    def scores(result):
        """
        calculate the model accuracy and error rate.
        function compares the actula class label versus predicted class label
        at each index. It sums all the true comparison results and divide by the 
        number of observations.
        @return model scores
        """  
        if result is None:
            raise ValueError( "The parameter 'data' must be assigned a non-nil reference to a Pandas DataFrame")
            
        #save model scores
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
        metrics["sensitivity"] = tp/(tp+fn); metrics["precision"]= tp/(tp+fp);  metrics["specificity"] = tn/(tn+fp)
        metrics["F_1"] = (2*metrics["sensitivity"]*metrics["precision"])/(metrics["sensitivity"] + metrics["precision"])
                
        return  metrics

    def confusion_matrix(result):
        """
        the function counts the true postives, true negavtives, false postives, and false negatives
        @return confusion matrix table
        """
        metrics = {}
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
        metrics["tp"] = tp;  metrics["tn"] = tn;  metrics["fn"] = fn;  metrics["fp"] = fp                    
        cmatrix = tabulate([["Predicted 1", tp, fp], ["Predicted 0", fn, tn]], headers=["", "Actual 1", "Actual 0"])
                
        return metrics

