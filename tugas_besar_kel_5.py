import streamlit as st
import pandas as pd 
import numpy as np 
from PIL import Image
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load

def extract_features(filename, model):
        try:
            image = Image.open(filename)
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'result: '
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


pilihan = st.sidebar.selectbox('Pilih Menu',('Beranda','Image Caption.Beta','Tentang Kami'))
if pilihan == 'Beranda':
    st.header('Selamat Datang di Web Apps Kelompok 5')
    st.subheader('Tugas Besar Mata Kuliah Associate Data Scientist')
    st.image('https://i3.wp.com/untirta.ac.id/wp-content/uploads/2020/12/logo-kampus-merdeka.png')

if pilihan == 'Image Caption.Beta':
    st.subheader('Silahkan Upload Gambar Yang Akan di Prediksi') 
    image_file = st.file_uploader('Unggah Gambar',type = ['png','jpg','jpeg'])   
    img = Image.open(image_file)
    max_length = 32
    tokenizer = load(open("C:/Users/Client/tokenizer.p","rb"))
    model = load_model('C:/Users/Client/model_9.h5')
    if image_file is not None: 
        xception_model = Xception(include_top=False, pooling="avg") 
        photo = extract_features(image_file, xception_model)
        description = generate_desc(model, tokenizer, photo, max_length)   
        file_details = {'filename':image_file.name,'filetype':image_file.type,'filesize':image_file.size} #melihat atribut image
        st.write(file_details) # Menampilkan detail file yang di upload
        st.image(img) # Menampilkan gambar yang diupload 
        st.write(description)


if pilihan == 'Tentang Kami':
    """
    ## Aji Begawan
    ## Farouk Abdul Royan
    ## Goldian
    """