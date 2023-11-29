from django.shortcuts import render
from .forms import ImageForm
from tensorflow.keras.preprocessing import image
from keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def predictor(request):
    
    classification = ""
    
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        print(f"post req: {request.POST}")
        print(f"files: {request.FILES}")
 
        if form.is_valid():
            face_image = form.save()
            image_path = face_image.Image.path
 
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            
            face_cascade = cv2.CascadeClassifier('./savedModels/haarcascade_frontalface_default.xml') 

            faces = face_cascade.detectMultiScale(img, 1.3, 4)
            print('Number of detected faces:', len(faces))
            
            # fix more than one faces later when you get time
            if len(faces) > 0:
                for i, (x, y, w, h) in enumerate(faces):
                    face = img[y:y + h, x:x + w]
            
            
            face = cv2.resize(face, (120, 120))
            
            img_array = image.img_to_array(face)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            
            json_file = open("./savedModels/affect_model.json", "r")
            loaded_model_json = json_file.read()
            json_file.close()

            loaded_model_v = model_from_json(loaded_model_json)
            loaded_model_a = model_from_json(loaded_model_json)

            loaded_model_v.load_weights("./savedModels/valence-weights-improvement-137-0.23.h5")
            loaded_model_a.load_weights("./savedModels/arousal-weights-improvement-95-0.17.h5")

            loaded_model_v.compile(loss='mean_squared_error', optimizer='sgd')
            loaded_model_a.compile(loss='mean_squared_error', optimizer='sgd')

            prediction_v = loaded_model_v.predict(img_array)
            prediction_a = loaded_model_a.predict(img_array)
    
            str_prediction_v = str(prediction_v[0][0])
            str_prediction_a = str(prediction_a[0][0])

            classification = f"Valence: {str_prediction_v} and Arousal: {str_prediction_a}"
            os.remove(image_path)
            face_image.delete()
   
    else:
        form = ImageForm()
        
    return render(request, 'main.html', {'form': form, 'result' : classification})


    