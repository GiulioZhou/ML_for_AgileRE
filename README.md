---------------------------------
1- Run the project
---------------------------------

1-Download a doc2vec model (https://github.com/jhlau/doc2vec) and unpack it in the folder
2-run the file script.py
3-follow the instructions to select the target, the features and the cardinality of the training set
4-The predictions will be saved in the partial results folder

Run fast_test_script.py if you prefer to skip the interactive part, but you have to modify the code in order to select the features.

NOTE: you need to manually comment the PCA and feature selection functions in dataExtractor.py if you don't want to use them.

---------------------------------
2- Dataset
---------------------------------

The dataset needs to be in the .csv format.
It is composed as follows:
1- User Stories (String)
2- Business Value (String)
3- User Story Elaboration (String)
4- Definition of done (String)
5- Expected output (String)
6- LOC (int)
7- New classes (int)
8- Changed classes (int)
9- Effort (int)
10- N. Unit Test (int)
11- Entropy (int)
12- Services (Matrix of int)

