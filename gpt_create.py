from openai import OpenAI
from gpt_utils import load_paired, ensure_directory_exists, save_result
from main import get_args
import numpy as np
import base64
import time
import os

with open('apikey.txt', 'r') as file:
    oai_api_key = file.read().strip()

client = OpenAI(
    api_key=oai_api_key,
    base_url="https://cmu.litellm.ai",
)

initial_prompt = """You are an expert in analyzing histological images with a deep understanding of the underlying biological processes and physical characteristics of cellular structures.
        I will provide you with an image that includes the original whole slide image on the left and the corresponding colored segmentation mask overlaid onto the original whole slide image on the right.
        In this segmentation mask, each color represents a different cell type: black for neoplastic cells, red for inflammatory cells, green for connective/soft tissue cells, blue for dead cells, yellow for epithelial cells, and cyan for the background.

        Your task is to analyze both the biological processes and physical attributes (large focus on physical attributes in the specific image) in the tissue based on the segmentation mask in 150 words MAXIMUM. Specifically:

        Describe the distribution, relative density, and spatial arrangement of each cell type with respect to the image. Essentially where are what cells located in the image. 
        Note the typical shapes and sizes of cells in each category, especially where clustering or unusual formations occur.
        Observe any color variations or gradients in the original image that may suggest changes in cell composition, health, or metabolic activity.
        Context: I am creating a segmentation model and want to incorporate textual features to improve segmentation accuracy. Again you have 150 words to write this. Do not go over."""
store_dict = {}
inst = 0

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f'Error encoding image {image_path}')
        return None

def forward(img_path):
    messages = [
    {"role": "system", 
        "content": initial_prompt}]

    next_prompt = f"Please analyze the image and provide a detailed explanation of the biological processes occurring in the tissue based on the image and segmentation mask. The labels can be found in the legend of each image and determine which cells/classes are in the image."
    base64_image = encode_image(img_path)
    messages.append(
                    {'role': 'user', 
                    'content': [
                        {'type': 'text', 'text': next_prompt},
                        {'type': 'image_url', 'image_url':
                            {'url': f"data:image/png;base64,{base64_image}"}}
                    ]}
                )

    response = client.chat.completions.create(
        model = 'gpt-4o',
        messages = messages,
        temperature = 0.0, # higher temp means more randomness, lower temp means more deterministic
        max_tokens = 200
    )
    messages.append({'role': 'assistant',
                        'content': f'{response.choices[0].message.content}'})
    # print(response.choices[0].message.content)

    store_dict[inst] = {
    'response': response.choices[0].message.content,
        'next_prompt': next_prompt,
        'img_path' : img_path
    }
    time.sleep(3) 
    return store_dict

def main():
    image_files, text_files = load_paired('./Data/Data/joined_images/', './Data/Data/texts/')

    for image, text_path in zip(image_files[431:], text_files[431:]):
        print(f'Image File: {image} Text File Saved To: {text_path}')

        start_time = time.time()
        # generate text
        text = forward(image)
        end_time = time.time()
        print(f'Durtion for generating text: {end_time - start_time}')

        # save it 
        with open(text_path, 'w') as file:
            file.write(text[inst]['response']) 


if __name__ == "__main__":
    main()

