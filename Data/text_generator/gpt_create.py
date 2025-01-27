from openai import OpenAI
from gpt_utils import load_paired, ensure_directory_exists, save_result
import numpy as np
import base64
import time
import os
from gpt_utils import ensure_directory_exists


with open('apikey.txt', 'r') as file:
    oai_api_key = file.read().strip()

# print(repr(oai_api_key))

client = OpenAI(
    api_key=oai_api_key,
)

initial_prompt_1 = """You are an expert in analyzing histological images with a deep understanding of the underlying biological processes and physical characteristics of cellular structures.
        I will provide you with an image that includes the original whole slide image on the left and the corresponding colored segmentation mask overlaid onto the original whole slide image on the right.
        In this segmentation mask, each color represents a different cell type: black for neoplastic cells, red for inflammatory cells, green for connective/soft tissue cells, blue for dead cells, yellow for epithelial cells, and cyan for the background.

        Your task is to analyze both the biological processes and physical attributes (large focus on physical attributes in the specific image) in the tissue based on the segmentation mask. Specifically:

        Describe the distribution, relative density, and spatial arrangement of each cell type with respect to the image. Essentially where are what cells located in the image. 
        Note the typical shapes and sizes of cells in each category, especially where clustering or unusual formations occur.
        Observe any color variations or gradients in the original image.
        Context: I am creating a segmentation model and want to incorporate textual features to improve segmentation accuracy. You have 75 words to generate this, use brief language"""

initial_prompt_2 = '''You are an expert pathologist whose eye catches both the broad patterns and minute details in tissue specimens.
                    I'll provide composite images: original whole slide (left) and segmentation overlay (right). The overlay distinguishes:

                    Black: Neoplastic cells
                    Red: Inflammatory cells
                    Green: Connective/soft tissue cells
                    Blue: Dead cells
                    Yellow: Epithelial cells
                    Cyan: Background

                    In 75 words, analyze this specific sample with rich detail:

                    The tissue's cellular landscape: How do different cell populations arrange themselves? Note areas of dense clustering, sparse regions, and interfaces between cell types
                    Morphological features: Describe each cell type's distinctive shapes (elongated, rounded, irregular), sizes (varying or uniform), and any striking patterns or formations
                    Visual indicators: Examine subtle color variations, intensity differences, and textural changes that might reveal cellular activity, stress, or tissue health

                    Context: Your detailed observations will help train an AI model to better segment these features. Focus on the physical characteristics unique to this specimen.
                    '''

initial_prompt = initial_prompt_2

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

    # next_prompt_1 = f"Provide analysis on where cells are located in the image. Describe in detail the shape of the cells. And focus on what patterns in the original image correlate to a specific type of cell/mask to make classifying them easier. The labels can be found in the legend of each image. DO IT IN LESS THAN 75 WORDS, USE BULLET POINTS. DO NOT talk about cells which are not in the image"
    next_prompt_1 = '''Analyze this histological image in under 75 words:

                        Map the precise location of each visible cell type
                        Describe distinctive cell morphology (shape, size, borders)
                        Note key visual patterns in the original image that help identify each cell type in the segmentation mask

                        Focus only on cells present in this specific sample. Refer to the image legend for cell type identification. Limit response to 75 words.'''
    
    next_prompt = next_prompt_1
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
        max_tokens = 100
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

    text_path = '../Data/texts/' if initial_prompt==initial_prompt_1 else '../Data/texts_2/'
    print(text_path)
    # input("Enter Something")
    image_files, text_files = load_paired('../Data/joined_images/', text_path)
    print(len(image_files))
    print(len(text_files))
    for image, text_path in zip(image_files[:], text_files[:]):
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