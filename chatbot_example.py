# To download the model and run inference locally:
#from transformers import pipeline
#pipe = pipeline("text-generation", model="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5")

# To use the huggingface-hosted model:
import requests
from access_token import HF_TOKEN

# The url to the model - should be in the model page on huggingface.
CHAT_API_URL = "https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"
CLASSIFIER_API_URL = "https://api-inference.huggingface.co/models/sagarnildass/suicide_detector_roberta_base"

# To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens)
headers = {"Authorization": f"Bearer {HF_TOKEN}"}


def query(payload, api_url):
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()


output = query({
    "inputs": "I am feeling sad",
}, CHAT_API_URL)


output = query({
    "inputs": "<|prompter|>I am feeling sad<|endoftext|><|assistant|>",
}, CHAT_API_URL)


def get_assistant_response(user_text):
    output = query({"inputs": f"<|prompter|>{user_text}<|endoftext|><|assistant|>",}, CHAT_API_URL)
    return output[0]['generated_text'].split("<|assistant|>")[1]

############### part 2: suicide detection #############################
# output = query({"inputs": "I am feeling sad",}, CLASSIFIER_API_URL)
# output = query({"inputs": "I am feeling happy",}, CLASSIFIER_API_URL)
# output = query({"inputs": "How to kill myself",}, CLASSIFIER_API_URL)


def is_suicidal(text):
    output = query({"inputs": text}, CLASSIFIER_API_URL)
    label_to_text = {'LABEL_0': 'suicide risk', 'LABEL_1': 'not suicide'}
    main_label = output[0][0]
    return label_to_text[main_label['label']], main_label['score']


def classify_then_answer(user_text):
    label, score = is_suicidal(user_text)
    if label == 'suicide risk':
        return "Please call 1201 or use the link: https://www.eran.org.il/ for emotional support."
    return get_assistant_response(user_text)