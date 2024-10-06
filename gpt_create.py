from openai import OpenAI
import base64
import time

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

with open('./.huggingface/api_keys.txt', 'r') as file:
        file_contents = file.readlines() ### I WILL PROVIDE THIS API KEY
        
oai_api_key = file_contents[1]
print(oai_api_key)

client = OpenAI(api_key = oai_api_key)

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

while True:
    labels = str(input("Enter the labels for the image: "))
    next_prompt = f"Please analyze the image and provide a detailed explanation of the biological processes occurring in the tissue based on the image and segmentation mask. There are {labels} present in the image."
    img_path = str(input("Enter the path to the image: "))

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
    print(response.choices[0].message.content)
    time.sleep(5)  # Sleep for 5 seconds before sending the next prompt
    
    # store messages, img, labels, and response in a dictionary
