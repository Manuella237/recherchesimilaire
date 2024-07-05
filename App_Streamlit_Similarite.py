import streamlit as st
import cv2
import face_recognition
import streamlit as st
import pandas as pd
import numpy as np
from App_login import main as mainLogin
import  warnings
from distances import retireve_similar_image
from descriptors import glcm,haralick, haralick_glcm
from PIL import Image
import os

def loadSignature(signature):
    if(signature=="GLCM"):
        signatures = np.load('./Signatures/signatures_glcm.npy')
    elif(signature=="HARALICK"):
        signatures = np.load('./Signatures/signatures_haralick.npy')
    elif(signature=="HARALICK ET GLCM"):
        signatures = np.load('./Signatures/signatures_haralick_glcm.npy')
    
    return signatures

def nameDescriptor(signature):
    if(signature=="GLCM"):
        name = glcm
    elif(signature=="HARALICK"):
        name = haralick
    elif(signature=="HARALICK ET GLCM"):
        name  = haralick_glcm
    
    return name


def load_image(image):
    img = Image.open(image)
    img_array = np.array(img)
    return img_array

def extract_features(image, carac):
    if image is not None:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = carac(gray_image)
        return features

def Acceuil():
    st.markdown("## Recherche d'images par similarité 🎈")
    st.write("### Choisir une image")   
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        img_array = load_image(uploaded_file)
        Descripteurs = st.radio("Descripteurs", ("GLCM", "HARALICK", "HARALICK ET GLCM"))
        Distance = st.radio("Distances", ("Euclidean", "Manhattan", "Chebyshev",  "Canberra"))
        nombre_images = st.number_input("Nombre d'images à afficher", min_value=1, value=4)
        signature = loadSignature(Descripteurs)
        nameDesc = nameDescriptor(Descripteurs)
        features = extract_features(img_array, nameDesc)
        result = retireve_similar_image(feature_db=signature, query_features=features, distance=f'{Distance}', num_results=nombre_images)
        st.subheader("Similar Images")
        image_width = 200
        images_per_row = 3
        num_rows = (len(result) + images_per_row - 1) // images_per_row
        for row in range(num_rows):
            cols = st.columns(images_per_row)
            for i in range(images_per_row):
                index = row * images_per_row + i
                if index < len(result):
                    img_path, dist, label = result[index]
                    img_path = os.path.join('datasets', img_path)
                    img = cv2.imread(img_path)
                    resized_img = cv2.resize(img, (image_width, image_width))
                    cols[i].image(resized_img, caption="Distance: {:.2f}".format(dist), width=image_width, use_column_width=False)
                    cols[i].write("Label: {}".format(label))
        # Affichage de lhistogramme 
        label_counts = {}
        for img_path, _, _ in result:
            # Extraire le nom du dossier parent
            parent_folder = os.path.basename(os.path.dirname(img_path))
            if parent_folder in label_counts:
                label_counts[parent_folder] += 1
            else:
                label_counts[parent_folder] = 1

        st.subheader("Histogramme")
        st.bar_chart(label_counts)

       
page_names_to_funcs = { 
    "Login": mainLogin,
    "Recherche d'Image similaire": Acceuil
}

selected_page = st.sidebar.selectbox("Pages", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()