import streamlit as st
import cv2
import face_recognition
import numpy as np
from PIL import Image
import os

signatures_class = np.load('FaceSignatutes_Recherches.npy') 
X = signatures_class[:, :-1].astype('float')
Y = signatures_class[:, -1]


def load_image(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1) 
    small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)  
    return small_frame

def load(image):
    img = Image.open(image)
    img_array = np.array(img)
    return img_array



def main():
    st.markdown("## Recherche de mes Images 🎈")
    st.write("### Choisir une image")   
    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png", "bmp"])
 
    
    if uploaded_file is not None:
        query_image = load_image(uploaded_file)
        im= load(uploaded_file)
        st.image(uploaded_file, caption="Image sélectionnée", width=200)
        
        # Encode le visage de l'image téléchargée
        query_encodings = face_recognition.face_encodings(query_image)

        
        # Vérifie s'il y a des visages dans l'image téléchargée
        if len(query_encodings) > 0:
            query_encoding = query_encodings[0]  # Prend le premier visage trouvé
            # Recherche des images similaires dans X
            similar_images = []
            for i, image_encoding in enumerate(X):
                matches = face_recognition.compare_faces([image_encoding], query_encoding)
                if matches[0]:
                    face_distances = face_recognition.face_distance([image_encoding], query_encoding)
                    best_match_index = np.argmin(face_distances)
                    similar_images.append((Y[i], face_distances[best_match_index]))

        
            
            if similar_images:
                st.subheader("Images similaires")
                image_width = 200
                images_per_row = 3
                num_rows = (len(similar_images) + images_per_row - 1) // images_per_row
                for row in range(num_rows):
                    cols = st.columns(images_per_row)
                    for i in range(images_per_row):
                        index = row * images_per_row + i
                        if index < len(similar_images):
                            label, distance = similar_images[index]
                            img_path = os.path.join('./Images/Recherches', f'{label}.jpg')

                            similar_img = Image.open(img_path)
                            if similar_img is not None:
                                cols[i].image(similar_img, caption=f"Nom: {label}, Distance: {distance:.2f}", width=image_width, use_column_width=False)
                            else:
                                st.write(f"Échec de chargement de l'image à l'index {index}.")
            else:
                st.write("Aucune image similaire trouvée dans la base de données.")
        else:
            st.write("Aucun visage trouvé dans l'image téléchargée.")




if __name__ == "__main__":
    main()
