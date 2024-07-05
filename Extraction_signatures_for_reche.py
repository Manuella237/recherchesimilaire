import cv2
import numpy as np
import face_recognition
import os
path = './Images/Recherches'
images = []
classNames = [] 
myList = os.listdir(path)
for img in myList:
    curImg = cv2.imread(os.path.join(path, img))
    images.append(curImg)
    imgName = os.path.splitext(img)[0]
    classNames.append(imgName)

def findEncodings(img_List, imgName_List):
    signatures = []
    count = 1
    for myImg, name in zip(img_List, imgName_List):
        img = cv2.cvtColor(myImg, cv2.COLOR_BGR2RGB)
        signature = face_recognition.face_encodings(img)[0]
        signature_class = signature.tolist() + [name]
        signatures.append(signature_class)
        print(f'{int((count/(len(img_List)))*100)} % extracted ...')
        count += 1
    face_array = np.array(signatures)
    np.save('FaceSignatutes_Recherches.npy', face_array)
    print('Signature saved')

def main():
    findEncodings(images, classNames)

if __name__ == '__main__':
    main()
    