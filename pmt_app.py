import os
import torch
import csv
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm
# Path to your locally downloaded model
model_path = "llava-hf/llava-1.5-7b-hf"

# Load the model and processor
processor = AutoProcessor.from_pretrained(model_path)
model = LlavaForConditionalGeneration.from_pretrained(model_path)

def llava_model(prompt:str, image:Image):
    inputs = processor(text=prompt, images=image, return_tensors="pt")
                
    # Generate the output
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    
    # Decode the result
    result = processor.decode(outputs[0], skip_special_tokens=True)
    print("type of file",type(image))

    width, height = image.size

    print(width, height)

    return result
def pn_npn_classifier(image:Image):
    text = "Does this image contain any sexual activity, pornography, genitals, or nudity? Answer only with 'Yes' or 'No'."
    result = llava_model(text,image)
    #pn_npn_result = result.split(':')[1]
    return result
def csam_classifier(image:Image):
    text = "Does this image contain a child? Answer only with 'Yes' or 'No'."
    result = llava_model(text,image)
    #csam_result = result.split(':')[1]
    return result

