import streamlit as st
import torch
from PIL import Image
#from torchvision import transforms
from io import BytesIO
import base64
from pmt_app import llava_model
import time
# Set the page configuration for a wider layout and custom page title
st.set_page_config(page_title="LLaVA Model - Image & Prompt Classifier", layout="wide")



# sidebar for instructions

st.sidebar.header("Instructions")
st.sidebar.write("""
1. **Enter a prompt** describing what you'd like the model to analyze in the image.
2. **Upload an image**.
3. **Press Submit** to get the prediction.
""")

# Input layout: Use columns for a better visual structure
col1, col2 = st.columns(2)

with col1:
    st.subheader("Step 1: Enter a Prompt")
    prompt = st.text_input("Prompt", placeholder="Enter the prompt.")

    st.subheader("Step 2: Upload an Image")
    uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

    # Show a placeholder image if none is uploaded
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.write("Please upload an image to proceed.")

# Result display
with col2:
    st.subheader("Step 3: View Results")

    # Display a submit button for predictions
    if st.button("Submit"):
        if uploaded_image and prompt:
            with st.spinner("Wait until we get the result..."):

                result = llava_model(prompt,image)



            # Display the prediction result
            st.success(f"**Prediction:** {result}")
        else:
            st.error("Please enter a prompt and upload an image.")
    else:
        st.write("Click 'Submit' to see the results.")

# Footer
st.markdown("""
    <hr style="border-top: 2px solid gray;">
    <p style='text-align: center; font-size: small;'>Created by <strong>Rushikesh</strong> | Powered by Llava, Streamlit & PyTorch</p>
""", unsafe_allow_html=True)