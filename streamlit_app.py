import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# Initialize the model (make sure you have the correct model name)
model_id = "CompVis/stable-diffusion-v1-4"  # Replace with the appropriate model ID if needed
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")  # Use "cpu" if you don't have a CUDA device

def generate_image(input_image, text_prompt="lego blocks minifigurine"):
    # Generate the image based on the input image and the text prompt
    combined_prompt = f"{text_prompt} based on the provided image"
    
    # Generate the image
    image = pipe(combined_prompt, init_image=input_image, strength=0.75, guidance_scale=7.5).images[0]
    
    return image

# Streamlit UI
st.title("Text-to-Image Generator with Input Image")
st.write("Generate an image by combining the provided image with a non-editable text prompt.")

# Fixed text prompt
text_prompt = "lego blocks minifigurine"
st.write(f"Text Prompt: **{text_prompt}**")

# Upload input image
input_image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if input_image_file is not None:
    # Load the image
    input_image = Image.open(input_image_file).convert("RGB")
    
    # Display the input image
    st.image(input_image, caption="Input Image", use_column_width=True)
    
    # Generate and display the output image
    if st.button("Generate Image"):
        with st.spinner("Generating image..."):
            output_image = generate_image(input_image, text_prompt)
            st.image(output_image, caption="Output Image", use_column_width=True)
