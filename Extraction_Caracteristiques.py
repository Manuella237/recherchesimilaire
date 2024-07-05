import os
import cv2
from descriptors import glcm,haralick, haralick_glcm
import numpy as np



def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        features = glcm(img)
        return features
    else:
        pass

out_folder = "Signatures"

def process_datasets(root_folder):
    for descriptor in ['glcm', 'haralick', 'bitdesc', 'haralick_glcm', 'bit_glcm']:
        all_features = []
        print(f"Processing dataset for descriptor: {descriptor}")

        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.bmp', '.jpeg')):
                    relative_path = os.path.relpath(os.path.join(root, file), root_folder)
                    img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        features = None
                        if descriptor == 'glcm':
                            features = glcm(img)
                        elif descriptor == 'haralick':
                            features = haralick(img)
                        elif descriptor == 'haralick_glcm':
                            features = haralick_glcm(img)
                        if features:
                            class_name = os.path.basename(os.path.dirname(relative_path))
                            all_features.append(features + [class_name, relative_path])
                    else:
                        print(f"Failed to load image: {file}")

        if all_features:
            signatures = np.array(all_features)
            np.save(os.path.join(out_folder, f'signatures_{descriptor}.npy'), signatures)
            print(f'Data for descriptor "{descriptor}" stored successfully')

    print('All data stored successfully')

# Main function
def main():
    print('Process initiated ...')
    process_datasets('./datasets')

if __name__ == '__main__':
    main()


