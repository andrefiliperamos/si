class Node:

    def __init(self, feature_idx = None, threshold = None, left = None, right = None, info_gain = None, value = None):


        '''
        • parameters:
        - feature_idx – index of the feature in X
        - threshold – threshold for the node
        - left – left node
        - right – right node
        - info_gain – information gain for the specific split
        - value – predicted class (only used in leaf nodes)
        '''

        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

class decisionTreeClassifier:

