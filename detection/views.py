import numpy as np
from django.shortcuts import render
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import keras.models
from keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image# type: ignore
from PIL import Image
from .forms import ImageUploadForm, SymptomForm
import imghdr
import joblib
import pandas as pd


ct_scan_model = keras.models.load_model("mobnet_n_v_c.h5")
cancer_model = keras.models.load_model("mobnet_model_best.hdf5")
symptoms_model = joblib.load("model_symptoms.pkl")


def predict_lung_cancer_sym(request):
    if request.method == 'POST':
        form = SymptomForm(request.POST)
        if form.is_valid():
            # Get user input
            GENDER = form.cleaned_data['GENDER']
            # Get other symptom input fields from the form
            AGE = form.cleaned_data['AGE']
            SMOKING = form.cleaned_data['SMOKING']
            YELLOW_FINGERS = form.cleaned_data['YELLOW_FINGERS']
            ANXIETY = form.cleaned_data['ANXIETY']
            PEER_PRESSURE = form.cleaned_data['PEER_PRESSURE']
            CHRONIC_DISEASE = form.cleaned_data['CHRONIC_DISEASE']
            FATIGUE = form.cleaned_data['FATIGUE']
            ALLERGY = form.cleaned_data['ALLERGY']
            WHEEZING = form.cleaned_data['WHEEZING']
            ALCOHOL_CONSUMING = form.cleaned_data['ALCOHOL_CONSUMING']
            COUGHING = form.cleaned_data['COUGHING']
            SHORTNESS_OF_BREATH = form.cleaned_data['SHORTNESS_OF_BREATH']
            SWALLOWING_DIFFICULTY = form.cleaned_data['SWALLOWING_DIFFICULTY']
            CHEST_PAIN = form.cleaned_data['CHEST_PAIN']

            # Encode gender (as you did in your code)
            if GENDER == 'Male':
                gender_encoded = 0
            else:
                gender_encoded = 1

            # Prepare input data for prediction (e.g., create a DataFrame)
            # Create a dictionary with feature names and values
            data = {
                'GENDER': [gender_encoded],
                'AGE': [AGE],
                'SMOKING': [SMOKING],
                'YELLOW_FINGERS':[YELLOW_FINGERS],
                'ANXIETY': [ANXIETY],
                'PEER_PRESSURE': [PEER_PRESSURE],
                'CHRONIC DISEASE': [CHRONIC_DISEASE],
                'FATIGUE ': [FATIGUE],
                'ALLERGY ': [ALLERGY],
                'WHEEZING': [WHEEZING],
                'ALCOHOL CONSUMING': [ALCOHOL_CONSUMING],
                'COUGHING': [COUGHING],
                'SHORTNESS OF BREATH': [SHORTNESS_OF_BREATH],
                'SWALLOWING DIFFICULTY': [SWALLOWING_DIFFICULTY],
                'CHEST PAIN': [CHEST_PAIN],
            }

            # Create a DataFrame
            input_data = pd.DataFrame(data)

            # Make predictions with the loaded model
            class_probabilities = symptoms_model.predict(input_data)
            # Format the result message
            if class_probabilities[0] == 0:
                result_message = "No Lung Cancer"
            else:
                result_message = "Lung Cancer Detected"

            # Render a template with the results
            return render(request, 'sym_result_template.html', {
                'result_message': result_message,
                'class_probability': class_probabilities,
            })

    else:
        form = SymptomForm()
    return render(request, 'sym_prediction_form.html', {'form': form})


def process_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


def predict_ct_scan(image_path):
    classes_dir = ["ctscan", "normal"]
    img = Image.open(image_path)
    processed_image = process_image(img, target_size=(224, 224))
    preds = ct_scan_model.predict(processed_image) # type: ignore
    pred_index = np.argmax(preds)
    pred_class = classes_dir[pred_index]
    prob = round(np.max(preds) * 100, 2)
    return pred_class, prob


def predict_cancer(image_path):
    classes_dir = ["Adenocarcinoma", "Large cell carcinoma", "Normal", "Squamous cell carcinoma"]
    img = Image.open(image_path)
    processed_image = process_image(img, target_size=(224, 224))
    preds = cancer_model.predict(processed_image) # type: ignore
    pred_index = np.argmax(preds)
    pred_class = classes_dir[pred_index]
    prob = round(np.max(preds) * 100, 2)
    return pred_class, prob


def home(request):
    return render(request, 'index.html')


def home_async(request):
    context = {}
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']
            if not imghdr.what(image_file):
                context['error'] = 'Uploaded file is not a valid image file.'
                return render(request, 'home_async.html', context)
            # Save the image to the media folder
            file_path = default_storage.save('uploads/' + image_file.name, ContentFile(image_file.read()))
            image_url = default_storage.url(file_path)
            
            print("Image URL:", image_url)  # Debugging statement
            
            preds = predict_ct_scan(image_file)
            if preds[0] == "ctscan":
                cancer_pred = predict_cancer(image_file)
                context['result'] = {
                    'is_ct_scan': preds[0],
                    'ct_scan_pred': cancer_pred[1],
                    'cancer_pred': cancer_pred[0],
                }
            else:
                context['result'] = {
                    'is_ct_scan': preds[0],
                }
            context['image_path'] = image_url
    else:
        form = ImageUploadForm()
    context['form'] = form
    return render(request, 'home_async.html', context)
