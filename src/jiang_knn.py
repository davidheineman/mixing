# Adapted psuedocode from https://aclanthology.org/2023.findings-acl.426.pdf

import gzip
import numpy as np

def predict_knn_ncd(training_set, test_set, k=5):
    """ 
    Predict class labels using k-NN with Normalized Compression Distance (NCD). 
    
    training_set = [
        ("the cat sat on the mat", "animal"),
        ("dogs are friendly pets", "animal"),
        ("the stock market crashed", "finance"),
        ("invest in mutual funds", "finance")
    ]

    test_set = [
        ("the dog barked at the cat", None),
        ("stocks rose sharply today", None)
    ]

    predict_knn_ncd(training_set, test_set, k=3)
    >>> ['animal', 'finance']
    """
    predictions = []
    
    for x1, _ in test_set:
        # Compress test example
        Cx1 = len(gzip.compress(x1.encode()))
        
        # Calculate NCD to all training examples
        distances = []
        for x2, _ in training_set:
            Cx2 = len(gzip.compress(x2.encode()))
            x1x2 = " ".join([x1, x2])
            Cx1x2 = len(gzip.compress(x1x2.encode()))
            
            # NCD formula
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            distances.append(ncd)
            
        # Get top k nearest neighbors
        sorted_idx = np.argsort(np.array(distances))
        top_k_classes = [training_set[i][1] for i in sorted_idx[:k]]
        
        # Predict majority class
        pred = max(set(top_k_classes), key=top_k_classes.count)
        predictions.append(pred)
        
    return predictions