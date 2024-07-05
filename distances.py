import numpy as np
from scipy.spatial import distance
def manhattan_distance(v1, v2):
    return np.sum(np.abs(np.array(v1).astype('float') - np.array(v2).astype('float')))

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((np.array(v1).astype('float') - np.array(v2).astype('float'))**2))

def chebyshev_distance(v1, v2):
    return np.max(np.abs(np.array(v1).astype('float') - np.array(v2).astype('float')))

def canberra_distance(v1, v2):
    return distance.canberra(v1, v2)

def retireve_similar_image(feature_db, query_features, distance, num_results):
    similar_images = []
    for instance in feature_db:
        features, label, img_path = instance[:-2], instance[-2], instance[-1]
        dist = None  
        if distance == 'Euclidean':
            dist = euclidean_distance(query_features, features)
        elif distance == 'Manhattan':
            dist = manhattan_distance(query_features, features)
        elif distance == 'Chebyshev':
            dist = chebyshev_distance(query_features, features)
        elif distance == 'Canberra':
            dist = canberra_distance(query_features, features)
        similar_images.append((img_path, dist, label))
    similar_images.sort(key=lambda x: x[1])
    return similar_images[:num_results]

