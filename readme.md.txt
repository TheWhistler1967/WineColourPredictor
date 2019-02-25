Uses white and red cvs from the Wine Quality Data Set 'http://www3.dsi.uminho.pt/pcortez/wine/'
I am using them to predict if a wine is white or red based on the various features.

On wine.py run, it will do the following in order:

Loads data from the red and white csv's. 
1599 red
4898 white

Splits data 0.2
6497 train
1300 validation

Compares accuracy of:
SVM
LogR
KNN
CART
LinD
GaNB

Predictions:
Selects best performing algorithm based on accuracy, and then uses it to make predictions on the validation set.



See output for specific results.

Optional: Visual data analysis: scatter plots (x vs y) and heat map.