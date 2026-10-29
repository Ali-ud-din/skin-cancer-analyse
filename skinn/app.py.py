import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("Skin Cancer Classification App")
st.write("Upload a skin image to determine the classification class.")

# Load the saved model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('skin_cancer_model.keras')

try:
    model = load_model()
    class_names = ['Acne and Rosacea Photos', 'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions', 'Atopic Dermatitis Photos', 'Cellulitis Impetigo and other Bacterial Infections', 'Eczema Photos', 'Exanthems and Drug Eruptions', 'Herpes HPV and other STDs Photos', 'Light Diseases and Disorders of Pigmentation', 'Lupus and other Connective Tissue diseases', 'Melanoma Skin Cancer Nevi and Moles', 'Poison Ivy Photos and other Contact Dermatitis', 'Psoriasis pictures Lichen Planus and related diseases', 'Seborrheic Keratoses and other Benign Tumors', 'Systemic Disease', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Urticaria Hives', 'Vascular Tumors', 'Vasculitis Photos', 'Warts Molluscum and other Viral Infections']
except Exception as e:
    st.error(f"Error loading model. Make sure 'skin_cancer_model.keras' is placed in the same directory: {e}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    st.write("Classifying...")
    
    # Preprocess image to match training parameters
    img = image.resize((224, 224))
    img_array = np.array(img)
    
    # Ensure 3 channels (RGB)
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:,:,:3]
        
    img_array = np.expand_dims(img_array, axis=0)
    
    # Run inference
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    st.success(f"Prediction: {predicted_class} (Confidence: {confidence:.2f}%)")
