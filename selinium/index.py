# import streamlit as st
# import pandas as pd
# import numpy as np


from app import app
import streamlit as st
import time
from PIL import Image
import base64
import io
import seaborn as sns
import matplotlib.pyplot as plt
@st.cache_data
def scrape_data(input):
     app(input_field)
st.title("Personality prediction")
input_field = st.text_input('Enter your text')
st.write('You entered:', input_field)



small_title_css = """
<style>
.small-title {
    font-size: 20px;
    font-weight: bold;
}
</style>
"""
# CSS to make the image circular

#Render the CSS and HTML in Streamlit

@st.cache_data
def scrape_data(input):
     data= app(input)
     
     image_path = "/content/scraper/"+data['fullName']+"/pdp.jpg"
     image = Image.open(image_path)

     # Convert the image to base64
     buffered = io.BytesIO()
    
     image.save(buffered, format="JPEG")
     img_str = base64.b64encode(buffered.getvalue()).decode()
     circle_image_css = f"""
     <style>
     .centered {{
         display: flex;
         justify-content: center;
         align-items: center;
         margin-top: 20px;
     }}

     .circle-img {{
         border-radius: 50%;
         width: 300px;
         height: 300px;
         object-fit: cover;
     }}
     </style>
     <div class="centered">
         <img src="data:image/jpeg;base64,{img_str}" class="circle-img">
     </div>
     <p style="text-align: center;">{data['fullName']} profile picture</p>
     """

     st.markdown(circle_image_css, unsafe_allow_html=True)


     col1, col2, col3 = st.columns([1, 2, 1])

     with col1:
         st.markdown('<p class="small-title">Followers</p>', unsafe_allow_html=True)
         st.write(data['followersCount'])
     with col3:
         st.markdown('<p class="small-title">Following</p>', unsafe_allow_html=True)
         st.write(data['followsCount'])

     st.title("Personality Scores")
     col1, col2, col3 = st.columns([1, 1, 1])
     with col1:
         personality = data['personality'][1]
         max_trait = max(personality, key=personality.get)
         for trait, score in personality.items():
          st.write(f"{trait}: {score:.2f}")
     st.markdown(f"""<p>{data['fullName']} has the <b>{max_trait}</b> personality!</p>""", unsafe_allow_html=True)
     prob_predictions=personality
     
     st.title('Biography')
     st.write(data['biography'])
     images=[]
     for i in range(len(data['captions'])):
         images.append({"image_path": "/content/scraper/"+data['fullName']+"/"+data['image_ids'][i]+".jpg", "caption": data['captions'][i]})

     # Function to display an image with a caption
     def display_image_with_caption(image_path, caption):
        image = Image.open(image_path)
        # Resize image to ensure uniformity
        image = image.resize((200, 200))  # You can adjust the size as needed
        st.image(image, caption=caption, width=200, output_format='JPEG')
    

     # Layout for the 3x3 grid
     st.title("Posts")
     rows = 3
     cols = 3

     # Generate the grid layout
     for i in range(rows):
         cols_elements = st.columns(cols)
         for j in range(cols):
             index = i * cols + j
             with cols_elements[j]:
                 if(index < len(images)):
                     display_image_with_caption(images[index]["image_path"], images[index]["caption"])


 # Combine the CSS and HTML and render it in Streamlit
if(input_field != None and input_field != ""):
     scrape_data(input_field)




