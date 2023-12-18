import random
import numpy as np
from si.data.dataset import Dataset
from si.models

class RandomForestClassifier:

    def __init__(self, n_estimators, max_features, min_sample_split, max_depth, mode, seed):

        '''
        parameters:
        - n_estimators - number of decision trees to use
        - max_features - maximum number of features to use per tree
        - min_sample_split- minimum samples allowed in a split
        - max_depth - maximumdepth of the trees
        - mode - impurity calculationmode (gini or entropy)
        - seed - random seed to use to assure reproducibility

        • estimated parameters:
        - trees - the trees of the random forest and respective features used for training (initialized as na empty list)
        '''

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []

        # fit - train the decision trees of the random forest
        def fit(self, n_features, dataset: Dataset, n_samples):
            
            '''
            1. Sets the random seed
            2. Defines self.max_features to be int(np.sqrt(n_features)) if None
            3. Create a bootsrap dataset (pick n_samples random samples from the dataset with replacement and self.max_features random features without replacement from the original dataset)
            4. Create and train a decision tree with the bootstrap dataset
            5. Append a tuple containing the features used and the trained tree
            6. Repeat steps 3, 4 and 5 for all trees in the forest
            7. Return itself (self)
            '''
            random.seed(self.seed)

            if self.max_features == None:

                self.max_features = np.sqrt(n_features)
            
            random.sample(0, len(dataset.X), n_samples)

            listaIndicesSamplesReplacement = np.random.choice(dataset.X, n_samples, True)

            listaIndicesSamplesNoReplacement = np.random.choice(dataset.X, n_samples, False)


            
            # predict - predicts the labels using the ensemble models
            def predict(self):

                '''

                '''
            # score - computes the accuracy between predicted and real labels
            def score(self):

                '''

                '''