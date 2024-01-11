from typing import Literal, Tuple, Union
from collections import Counter
import random
import numpy as np
from si.io.csv_file import read_csv
from si.metrics.accuracy import accuracy
from si.data.dataset import Dataset
from si.models.decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier:

    def __init__(self, n_estimators, max_features, min_sample_split, max_depth, seed, mode: Literal['gini', 'entropy'] = 'gini'):

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
        
        # 1. Sets the random seed

        random.seed(self.seed)

        # 2. Defines self.max_features to be int(np.sqrt(n_features)) if None

        if self.max_features == None:

            self.max_features = np.sqrt(n_features)
        # 6. Repeat steps 3, 4 and 5 for all trees in the forest
        for _ in range(self.n_estimators):
            # 3. Create a bootsrap dataset (pick n_samples random samples from the dataset with replacement and self.max_features random features without replacement from the original dataset) 
            indicesSamples = np.random.choice(np.array(np.size(dataset.X, axis = 0)), n_samples, True)

            indicesFeatures = np.random.choice(np.array(np.size(dataset.X, axis = 1)), n_features, False)

            bootstrapDatasetX = dataset.X[np.ix_(indicesSamples, indicesFeatures)]

            bootstrapDatasetY = dataset.y[np.ix_(indicesSamples)]
            
            if dataset.features is not None:

                bootstrapDatasetFeatures = dataset.features[np.ix_(indicesFeatures)]
        
            else:

                bootstrapDatasetFeatures = None
            
            bStrapDataset = Dataset(bootstrapDatasetX, bootstrapDatasetY, bootstrapDatasetFeatures, dataset.label)
            
            # 4. Create and train a decision tree with the bootstrap dataset
            dtc = DecisionTreeClassifier()
            dtc.fit(bStrapDataset)
            #dtc.print_tree(self.tree)
            #input()
            # 5. Append a tuple containing the features used and the trained tree

            self.trees.append((bootstrapDatasetFeatures, dtc))

        # 7. Return itself (self)
        
        return self
        
        # predict - predicts the labels using the ensemble models
    def predict(self):

        allPredictions = []

        # 1. Get predictions for each tree using the respective set of features

        for tree in self.trees:

          predictions = [DecisionTreeClassifier()._make_prediction(x, tree[1].tree) for x in tree[1].dataset.X]



        # 2. Get the most common predicted class for each sample
          allPredictions.append(Counter(predictions).most_common()[0][0])
          
        # 3. Return predictions
        return np.array(allPredictions)
            
    # score - computes the accuracy between predicted and real labels
    def score(self):

        #1. Get predictions using the predict method

        predictions = self.predict()

        #2. Computes the accuracy between predicted and real values

        real = []

        for tree in self.trees:

          real.append(Counter(tree[1].dataset.y).most_common()[0][0])


        return accuracy(np.array(real), predictions)
           
rfc = RandomForestClassifier(1, 4, 10, 10, 10, 'gini')

ficheiro = read_csv('C:\\Percurso Académico\\Mestrado em Bioinformática - UMinho\\2023-2024\\02 Disciplinas\\1º Semestre\\03 Sistemas Inteligentes para a Bioinformática\\0Base de Dados\\iris.csv', ',', True, True)

rfc.fit(3, ficheiro, 8)

for x in rfc.trees:
    
    x[1].print_tree(x[1].tree)

print()
print('predict')
print(rfc.predict())
print(rfc.score())    
