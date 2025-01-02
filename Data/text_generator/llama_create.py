import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import time
import os
from gpt_utils import load_paired

# Login to Hugging Face

with open('apikey.txt', 'r') as file:
    api_key = file.read().strip()

login(api_key)

model_id = "meta-llama/Llama-3.2-11B-Vision"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

initial_prompt = """You are an expert in analyzing histological images with a deep understanding of the underlying biological processes and physical characteristics of cellular structures.
I will provide you with an image that includes the original whole slide image on the left and the corresponding colored segmentation mask overlaid onto the original whole slide image on the right.
In this segmentation mask, each color represents a different cell type: black for neoplastic cells, red for inflammatory cells, green for connective/soft tissue cells, blue for dead cells, yellow for epithelial cells, and cyan for the background.

Your task is to analyze both the biological processes and physical attributes (large focus on physical attributes in the specific image) in the tissue based on the segmentation mask in 150 words MAXIMUM."""

store_dict = {}
print(store_dict)
inst = 0

def forward(img_path):
    try:
        # Open image
        image = Image.open(img_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        print(f"Processing image: {img_path}")
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")

        inputs = processor.image_processor(images=image, return_tensors="pt")
        text_inputs = processor.tokenizer(initial_prompt, return_tensors="pt", add_special_tokens=True)
        
        model_inputs = {
            "input_ids": text_inputs["input_ids"].to(model.device),
            "attention_mask": text_inputs["attention_mask"].to(model.device),
            "pixel_values": inputs["pixel_values"].to(model.device),
        }

        # Generate
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.6
        )
        
        # Decode
        response = processor.decode(outputs[0], skip_special_tokens=True)  # Only decode the new tokens
        print(f"Generated response: {response[:100]}...")
        
        
        time.sleep(3)
        return response
        
    except Exception as e:
        print(f"Error processing image {img_path}: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def main():
    image_files, text_files = load_paired('../Data/joined_images/', '../Data/texts/')

    for image, text_path in zip(image_files[:1], text_files[:1]):
        print(f'Image File: {image} Text File Saved To: {text_path}')

        start_time = time.time()
        text = forward(image)
        end_time = time.time()
        print(f'Duration for generating text: {end_time - start_time}')

        if text:
            print(f"Writing response to {text_path}")
            with open(text_path, 'w') as file:
                file.write(text)
        else:
            print(f"Failed to process {image}")

if __name__ == "__main__":
    main()