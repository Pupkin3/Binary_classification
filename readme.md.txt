# python-binary-classification
python implementation (used scikit-learn framework) of  binary classification model. Three diferent algorithms were considered: Decision Tree, Random Forest and Logistic Regression.

## Running the code
Run the code with the python interpreter: 

```
python
>>>from python_binary_classification import Model
>>>Model("adult.csv", "RandomForest")
```


You have to specify:
 + filename with train dataset
 + [optional] algorithm to use (DecisionTree, RandomForest, LogReg)
 Note: The last one attribute is considered to be a target attribute (no need to specify).