# About the Dataset

## Dataset Information:

The dataset was downloaded from the UCI Machine Learning Repository.

The two datasets are related to the red and white variants of the Portuguese wine "Vinho Verde". Reference [Cortez et al., 2009]. Due to privacy and logistic issues, only physicochemical variables (inputs) and sensory variables (the output) are available (e.g., no data on grape types, wine brand, wine selling price, etc.).

These data sets can be viewed as classification or regression tasks. The classes are ordered and unbalanced (e.g., there are many more normal wines than excellent or poor wines). Anomaly detection algorithms could be used to detect the few excellent or poor wines. In addition, we are not sure if all input variables are relevant. Therefore, it might be interesting to test feature selection methods.

Two data sets were combined and some values were randomly removed.

## Attribute Information:

For more information, read [Cortez et al., 2009].

Input variables (based on physicochemical tests):
1 - fixed acidity
2 - volatile acidity
3 - citric acid
4 - residual sugar
5 - chlorides
6 - free sulfur dioxide
7 - total sulfur dioxide
8 - density
9 - pH
10 - sulfates
11 - alcohol
Output variable (based on sensory data):
12 - quality (score between 0 and 10).

## Acknowledgments:

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.