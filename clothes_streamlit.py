
import streamlit as st
from ultralytics import YOLO
import PIL
import requests
from io import BytesIO
import cv2

st.set_page_config( page_title = "Object Detection using YOLOv8", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")
st.title("Clothes Detection using YOLOv8")

confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100
model_path = "best.pt"

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
    
source_radio = st.sidebar.radio("Select Source", ["IMAGE in locale", "Image in link"])

source_img = None

if source_radio == "IMAGE in locale":
    st.write("Classes : T-shirt,  Dress,  Jacket,  Pants,  Shirt,  Short,  Skirt,  Sweater")
    source_img = st.sidebar.file_uploader("Choose an image...", type = ("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str("trendyol.png")
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image", use_column_width=True)
                
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image", use_column_width=True)
                
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str("trendyol_detected.png")
            default_detected_image = PIL.Image.open(default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image', use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")
                    
else:
    st.write("Link uzerinden tahmin yapmak istiyorsaniz; resme sag tiklayip *Resim adresini kopyala* kismindan url yi almaniz ve Sidebar'a kopyalamaniz gerekmektedir.")
    image_url = st.sidebar.text_input("Give an image Link...")
    col1, col2 = st.columns(2)
    with col1:
        
        try:
            response = requests.get(image_url)
            image = PIL.Image.open(BytesIO(response.content))

            st.image(image, caption='', use_column_width=True)
        except Exception as e:
            st.error("There was an error uploading the image. Please enter a valid internet link.")
            
    with col2:
        if st.sidebar.button('Detect Objects'):
                res = model.predict(image, conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")
