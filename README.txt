Classes:
MostCommonWords.py: Goes through all the given text and in each text finds the 100 most common words
                    that will be used as features.

features.py: Assigns the features to the text sample, each of the words found in MostCommonWords is a feature
             and we will assign 1 if its in the text and 0 if it's not.

preprocessor.py: Preprocess all the raw data into a design matrix.

model.py: Builds and trains a model

****See project.pdf for information on the construction of the model and the features choice