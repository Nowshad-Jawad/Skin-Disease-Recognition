from django.shortcuts import render
from django.conf import settings
from .forms import ImageForm
from .models import Image
from django.core.files.storage import FileSystemStorage
from skimage import img_as_float
from skimage.filters import gaussian
from skimage.restoration import denoise_bilateral

from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
from tensorflow import Graph
import json
import numpy as np
import random

img_height, img_width = 64, 64

actinic_keratosis=['An actinic keratosis sometimes disappears on its own but might return after more sun exposure.',
                   'It is hard to tell which actinic keratoses will develop into skin cancer, so they are usually removed as a precaution',
                  'Actinic keratoses can be removed by freezing them with liquid nitrogen.',
                  'If you have several actinic keratosis, your doctor might prescribe a medicated cream or gel to remove them, such as fluorouracil, imiquimod, ingenol mebutate or diclofenac.']

basal_cell_carcinoma=['Basal cell carcinoma is most often treated with surgery to remove all of the cancer and some of the healthy tissue around it.',
                     'The goal of treatment for basal cell carcinoma is to remove the cancer completely.',
                     'Very rarely, basal cell carcinoma may spread (metastasize) to nearby lymph nodes and other areas of the body.']

melanoma=['Types of surgery used to treat local and regional melanoma are wide excision, lymphatic mapping and sentinel lymph node biopsy, and lymph node dissection.',
         'The main treatment for melanoma is surgical removal, or excision, of the primary melanoma on the skin.',
         'If melanoma that has spread and causes symptoms, such as bone pain or headaches, then radiation therapy may help relieve those symptoms.',
         'The types of systemic therapies used for melanoma include: 1.Immunotherapy, 2.Targeted therapy and 3.Chemotherapy']

melanocytic_nevi=['A melanocytic naevus (American spelling ‘nevus’), or mole, is a common benign skin lesion due to a local proliferation of pigment cells (melanocytes).',
                 'A melanocytic naevus can be present at birth (a congenital melanocytic naevus) or appear later (an acquired naevus).',
                 'About 1% of individuals are born with one or more congenital melanocytic naevi. This is usually sporadic, with rare instances of familial congenital naevi.',
                 'Fair-skinned people tend to have more melanocytic naevi than darker skinned people.',
                 'Melanocytic naevi that appear during childhood (aged 2 to 10 years) tend to be the most prominent and persistent throughout life.',
                 'Melanocytic naevi that are acquired later in childhood or adult life often follow sun exposure and may fade away or involute later.']

benign_keratosis = ['Be careful not to rub, scratch or pick at it. This can lead to itching, pain and bleeding.',
                   'Cryosurgery can be an effective way to remove a benign keratosis.',
                   ' If you have a raised growth, your doctor may prescribe a solution of 40% hydrogen peroxide, which is applied to the skin.',
                   'Treatment may result in discoloration of treated skin']

dermatofibroma = ['Dermatofibromas are small, harmless growths that appear on the skin.',
                 'Dermatofibromas are harmless growths within the skin that usually have a small diameter. They can vary in color, and the color may change over the years.',
                 'Dermatofibromas are firm to the touch. They are very dense, and many people say they feel like a small stone underneath or raised above the skin.',
                 'Most dermatofibromas are painless.',
                 'When pinched, a dermatofibroma will not push towards the surface of the skin. Instead, it will dimple inward on itself, which can help tell the difference between a dermatofibroma and another type of growth.']

vascular_lesions = ['Vascular lesions are relatively common abnormalities of the skin and underlying tissues, more commonly known as birthmarks.',
                   'Vascular lesions in childhood are comprised of vascular tumors and vascular malformations.',
                   'There are three major categories of vascular lesions: Hemangiomas, Vascular Malformations, and Pyogenic Granulomas.']



model_graph =Graph()
with model_graph.as_default():
    tf_session=tf.compat.v1.Session()
    with tf_session.as_default():
        model=load_model('./models/cnn.h5')


# Create your views here.
def home(request):
    """if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()"""
    form = ImageForm()
    return render(request,'myapp/home.html', {'form':form})


def predict(request):
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
    img = Image.objects.last()
    qs=Image.objects.all()
    abs_url=str(img.photo.path)
    up_img = image.load_img(abs_url, target_size=(img_height, img_width))
    up_img=img_as_float(up_img)
    x = image.img_to_array(up_img)
    x=x/255
    x=x.reshape(1, img_height, img_width, 3)
    with model_graph.as_default():
        with tf_session.as_default():
            predi=model.predict(x)
    predi=np.argmax(predi)

    if predi == 0:
        predi = 'Actinic Keratoses'
        rx=random.choice(actinic_keratosis)
    elif predi == 1:
        predi = 'Basal Cell Carcinoma'
        rx=random.choice(basal_cell_carcinoma)
    elif predi == 2:
        predi= 'Benign Keratoses'
        rx=random.choice(benign_keratosis)
    elif predi == 3:
        predi = 'Dermatofibroma'
        rx=random.choice(dermatofibroma)
    elif predi == 4:
        predi = 'Melanoma'
        rx=random.choice(melanoma)
    elif predi == 5:
        predi = 'Melanocytic Nevi'
        rx=random.choice(melanocytic_nevi)
    elif predi == 6:
        predi = 'Vascular Lession'
        rx=random.choice(vascular_lesions)

    return render(request,'myapp/predict.html', {'img':img, 'form':form, 'predi':predi, 'rx':rx})
