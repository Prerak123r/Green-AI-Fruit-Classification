import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Set page config
st.set_page_config(page_title="Green AI: Fruit Ripeness", page_icon="🍏")

# 1. LOAD THE MODEL
@st.cache_resource
def load_my_model():
    # Loading the model you just saved
    model = tf.keras.models.load_model('fruit_ripeness_model.h5')
    return model

model = load_my_model()

# 2. DYNAMIC CLASS NAMES
# These must match the alphabetical order of your folders
class_names = [
    'Apple Overripe', 'Apple Ripe', 'Apple Unripe',
    'Banana Overripe', 'Banana Ripe', 'Banana Unripe',
    'Mango Overripe', 'Mango Ripe', 'Mango Unripe',
    'Orange Overripe', 'Orange Ripe', 'Orange Unripe',
    'Tomato Overripe', 'Tomato Ripe', 'Tomato Unripe'
]

# 3. PREDICTION FUNCTION
def predict_ripeness(img, model):
    size = (224, 224)    
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    # Normalize the image just like we did in Day 2
    img_scaled = img_array.astype(np.float32) / 255.0
    img_reshape = img_scaled[np.newaxis, ...]
    
    prediction = model.predict(img_reshape)
    return prediction

# 4. USER INTERFACE
st.title("🍏 Green AI: Multi-Fruit Ripeness Assistant")
st.write("Upload a photo of an Apple, Banana, Mango, Orange, or Tomato.")

# Replace your file_uploader line with this:
option = st.radio("Select Input Method:", ("Upload Image", "Use Camera"))

if option == "Upload Image":
    file = st.file_uploader("Choose a fruit image...", type=["jpg", "png", "jpeg"])
else:
    file = st.camera_input("Take a photo of the fruit")
    
if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner('AI is analyzing...'):
        predictions = predict_ripeness(image, model)
        # Get the index of the highest prediction
        result_index = np.argmax(predictions)
        label = class_names[result_index]
        confidence = np.max(predictions) * 100

    # Display Result
    st.subheader(f"Result: {label}")
    st.progress(int(confidence))
    st.write(f"**Confidence Score:** {confidence:.2f}%")

    # 5. EXPERT GUIDANCE (Yellow Hat - Practical Advice)
    if "Overripe" in label:
        st.error("⚠️ Overripe: Use immediately for smoothies, sauces, or baking!")
    elif "Unripe" in label:
        st.info("🕒 Unripe: Leave at room temperature for a few days to ripen.")
    else:
        st.success("✅ Ripe: Perfect for eating now! Store in the fridge to maintain freshness.")


    