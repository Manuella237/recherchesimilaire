import cv2
from skimage.feature import graycomatrix, graycoprops
import mahotas.features as features
#from bit import bio_taxo 
 
def glcm(image_data):
    try:
        glcm_matrix = graycomatrix(image_data, [2], [0], None, symmetric=True, normed=True)
        dissimilarity = graycoprops(glcm_matrix, 'dissimilarity')[0, 0]
        contrast = graycoprops(glcm_matrix, 'contrast')[0, 0]
        correlation = graycoprops(glcm_matrix, 'correlation')[0, 0]
        energy = graycoprops(glcm_matrix, 'energy')[0, 0]
        homogeneity = graycoprops(glcm_matrix, 'homogeneity')[0, 0]
        return [dissimilarity, contrast, correlation, energy, homogeneity]
    except Exception as e:
        print(f"An error occurred while extracting GLCM features: {e}")
        return None
 
def haralick(image_data):
    return features.haralick(image_data).mean(0).tolist()
 
#def bitdesc(image_data):
   # return bio_taxo(image_data)

def haralick_glcm(image_data):
    haralick_features = haralick(image_data)
    glcm_features = glcm(image_data)
    return haralick_features + glcm_features if haralick_features and glcm_features else None
 
#def bit_glcm(image_data):
    #bit_features = bitdesc(image_data)
    #glcm_features = glcm(image_data)
    #return bit_features + glcm_features if bit_features and glcm_features else None
 
#def bit_haralick(image_data):
    #bit_features = bitdesc(image_data)
    #haralick_features = haralick(image_data)
    #return bit_features + haralick_features if bit_features and haralick_features else None
 
 
