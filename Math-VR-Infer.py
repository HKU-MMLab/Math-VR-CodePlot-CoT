import json
import argparse
from datasets import load_dataset
import base64
import io
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    api_key="Your OpenAI API Key"
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_dir", type=str, default="./examples/answer.json", help="The path to save model's solution json file.")
    parser.add_argument("--data_path", type=str, default="./data/test-00000-of-00001.parquet", help="The path to the benchmark dataset.")
    parser.add_argument("--type", type=str, default="both", help="The type of model to use: text, multimodal or both.")
    args = parser.parse_args()
    return args

def img2base64(img, format="JPEG"):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=format)
    return base64.b64encode(img_byte_arr.getvalue()).decode()

def create_message(question, images=None):
    message = [
      {
          'role': 'user',
          'content': [
            {'type': 'text', 'text': question},
          ]
      }
    ]
    for image in images:
        message[0]['content'].append({'type': 'image_url', 'image_url': {"url": f"data:image/jpeg;base64,{img2base64(image)}"}})
    return message

args = parse_args()
dataset = load_dataset("parquet", data_files={"test": args.data_path})["test"]
answers = {}

for data in tqdm(dataset):
    question = data["question"]
    images = []
    for i in range(1, 9):
        if f"<image{i}>" in question:
            question = question.replace(f"<image{i}>", "")
            images.append(data[f"image{i}"])

    if (args.type == "text" and data["category"] == "multimodal") or (args.type == "multimodal" and data["category"] == "text"):
        continue
    
    message = create_message(question, images)
    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=message
    )
    answer = completion.choices[0].message.content
    answers[data["id"]] = answer
    with open(args.answer_dir, 'w') as f:
        json.dump(answers, f, indent=4)