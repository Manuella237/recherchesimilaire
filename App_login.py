import streamlit as st
import cv2
import face_recognition
import numpy as np

# Chargez les signatures faciales
signatures_class = np.load('FaceSignatutes.npy')
X = signatures_class[:, 0:-1].astype('float')
Y = signatures_class[:, -1]

def main():
    st.title("Login par Reconnaissance Faciale")
    start_button = st.button("Démarrer la reconnaissance faciale")

    if start_button:
        # Commencer la capture vidéo
        frame_window = st.image([])
        cap = cv2.VideoCapture(0)

        last_known = None 

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Erreur de capture vidéo.")
                break
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(X, face_encoding)
                face_distances = face_recognition.face_distance(X, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = Y[best_match_index]
                    if last_known != name: 
                        st.success(f'Connecté en tant que: {name}')
                        last_known = name  
                        if st.button('Stop reconnaissance'):
                            st.experimental_set_query_params(page="Acceuil")
                            st.experimental_rerun()
                else:
                    if last_known != "Inconnu": 
                        st.error("Visage non reconnu.")
                        last_known = "Inconnu"  
                break  
            frame_window.image(frame)

        cap.release()

if __name__ == "__main__":
    main()
