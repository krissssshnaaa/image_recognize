import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import matplotlib.pyplot as plt
import os
import random

def load_model_and_dicts(model_type):
    if model_type == 'MODEL-PCAT':
        model_path = './MODEL-PCAT/my_model.keras'
        wordtoix_path = './MODEL-PCAT/wordtoix.pkl'
        ixtoword_path = './MODEL-PCAT/ixtoword.pkl'
        max_length = 34
    else:
        model_path = './MODEL-QDOG/image-caption-30k-39.h5'
        wordtoix_path = './MODEL-QDOG/wordtoix.pkl'
        ixtoword_path = './MODEL-QDOG/ixtoword.pkl'
        max_length = 74

    model = load_model(model_path)
    
    with open(wordtoix_path, 'rb') as f:
        wordtoix = pickle.load(f)

    with open(ixtoword_path, 'rb') as f:
        ixtoword = pickle.load(f)

    return model, wordtoix, ixtoword, max_length

inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

def preprocess_img(img_path):
    img = load_img(img_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_features(photo):
    feature_vector = inception_model.predict(photo)
    return np.reshape(feature_vector, (1, feature_vector.shape[1]))

def greedy_search(features, model, wordtoix, ixtoword, max_length):
    start_seq = 'startseq'
    for _ in range(max_length):
        seq = [wordtoix[word] for word in start_seq.split() if word in wordtoix]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([features, seq], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        start_seq += ' ' + word
        if word == 'endseq':
            break
    return ' '.join(start_seq.split()[1:-1])

def load_descriptions():
    with open('./test_images/Training set descriptions.txt', 'r') as f:
        return f.readlines()

descriptions = load_descriptions()

st.markdown(
    """
    <style>
    .title {
        color: #2E86C1;
        text-align: center;
        font-size: 2em;
    }
    .caption {
        color: white;
        font-size: 1.5em;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 class='title'>Image Captioning</h1>", unsafe_allow_html=True)

model_type = st.selectbox("Select Model", ['MODEL-PCAT', 'MODEL-QDOG'], key='model_selection')

temp_dir = "tempDir"
os.makedirs(temp_dir, exist_ok=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility='visible')

if uploaded_file is not None:
    img_path = os.path.join(temp_dir, uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    model, wordtoix, ixtoword, max_length = load_model_and_dicts(model_type)

    photo = preprocess_img(img_path)
    features = extract_features(photo)
    caption = greedy_search(features, model, wordtoix, ixtoword, max_length)

    st.image(uploaded_file, caption='Uploaded Image', use_column_width='auto')
    st.markdown(f"<p class='caption'>Generated Caption: {caption}</p>", unsafe_allow_html=True)

if st.button('Training Set Descriptions'):
    random_descriptions = random.sample(descriptions, 10)
    for desc in random_descriptions:
        st.write(desc.strip())

