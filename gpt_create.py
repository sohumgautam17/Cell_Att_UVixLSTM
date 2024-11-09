from openai import OpenAI
from gpt_utils import load_paired, ensure_directory_exists, save_result
from main import get_args
import numpy as np
import base64
import time
import os

oai_api_key = os.getenv('OAI_API_KEY')  # Fetch the API key from the environment variable
print(oai_api_key)

client = OpenAI(
    api_key=oai_api_key,
    base_url="https://cmu.litellm.ai",
)

# system prompt (be exact)
initial_prompt = """You are an expert in analyzing histological images and have a deep understanding of the underlying biological processes.
I will provide you with an image that comprises of the original whole slide image on the left and the corresponding segmentation mask overlayed onto the original whole slide image on the right.
The segmentation mask is color-coded where black represents neoplastic cells, red represents inflammatory cells, green represents connective/soft tissue cells, blue represents dead cells, yellow represents epithelial cells, and turquoise represents background.
Your task is to analyze the image and provide a detailed explanation of the biological processes occurring in the tissue based on the segmentation mask.

Context: I am creating a segmentation model and I want to incroporate textual features in to the model to improve the segmentation accuracy.
"""

conversation_history = ""
messages = [
{"role": "system", 
    "content": initial_prompt}]

store_dict = {}
inst = 0
# while True:

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f'Error encoding image {image_path}')
        return None

def forward(img_path):

    # labels = str(input("Enter the labels for the image: ")) # The model can distinguish without the labels being told to it
    next_prompt = f"Please analyze the image and provide a detailed explanation of the biological processes occurring in the tissue based on the image and segmentation mask. The labels can be found in the legend of each image and determine which cells/classes are in the image."
    # There are {labels} present in the image.
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
        max_tokens = 1500
    )
    messages.append({'role': 'assistant',
                        'content': f'{response.choices[0].message.content}'})
    # print(response.choices[0].message.content)

    # store messages, img, labels, and response in a dictionary
    store_dict[inst] = {
    'response': response.choices[0].message.content,
        'next_prompt': next_prompt,
        'img_path' : img_path
    }
    time.sleep(2)  # Sleep for 5 seconds before sending the next prompt
    return store_dict

def main():
    image_files, text_files = load_paired('./Data/Data/joined_images/', './Data/Data/texts/')

    for image, text_path in zip(image_files, text_files):
        print(f'Image File: {image} Text File Saved To: {text_path}')

        # generate text
        text = forward(image)

        # save it 
        with open(text_path, 'w') as file:
            file.write(text[inst]['response'])  # Replace with GPT-generated text if available


if __name__ == "__main__":
    main()

