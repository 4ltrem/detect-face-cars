import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
car_cascade = cv2.CascadeClassifier('cars.xml') 
image = cv2.imread('image.jpg')
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
faces = face_cascade.detectMultiScale(grayImage)
cars = car_cascade.detectMultiScale(grayImage)
#print type(faces)
  
if len(faces) == 0:
    print ("Aucune détection faciale")

if len(cars) == 0:
    print ("Aucune détection automobile")

else:
    print (f'Faces: \n{faces}')
    print (f'Faces Shape: {faces.shape}')
    print ("Nombre de faces détectées: " + str(faces.shape[0]))
    
    print(f'Cars: \n{cars}\nCars Shape: {cars.shape}\nNombre de voiture détectées: {cars.shape[0]}')

    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
    
    for (x2,y2,w2,h2) in cars:
        cv2.rectangle(image,(x2,y2),(x2+w2,y2+h2),(0,255,0),1)

    cv2.rectangle(image, ((0,image.shape[0] -45)),(270, image.shape[0]), (255,255,255), -1)
    cv2.putText(image, "Nombre de faces détectées: " + str(faces.shape[0]), (0,image.shape[0] -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0,0,0), 1)
    cv2.putText(image, "Nombre de voiture détectées: " + str(cars.shape[0]), (0,image.shape[0] -30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0,0,0), 1)
    
    cv2.imshow('Image avec faces',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
