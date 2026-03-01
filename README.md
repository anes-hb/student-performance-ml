## Error Analysis

After evaluating the logistic regression model, i observed that most predictions are correct.  
However, some errors occur mainly for students whose final grade (G3) is close to the passing threshold (10).  
The model sometimes predicts that a student fails when they actually pass, and vice versa.  
Overall, the "pass" class is predicted more accurately than the "fail" class, likely due to class imbalance.