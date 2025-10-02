from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
from PIL import Image
import re

model_path = "./ckpts/MatPlotCode"

def save_image_from_response(response, image_filename):
    code_match = re.search(r"```python(.*?)```", response, re.DOTALL)
    if not code_match:
        code_match = re.search(r"```(.*?)```", response, re.DOTALL)
    code = re.sub(r"plt\.show\s*\(\s*\)", f'plt.savefig("{image_filename}")', code_match.group(1))
    try:
        exec(code, {}, {})
        return True, code_match.group(1)
    except Exception as e:
        
        return False, code_match.group(1)
    
def pad_to_square(image):
    width, height = image.size
    max_size = max(width, height)
    square_image = Image.new("RGB", (max_size, max_size), (255, 255, 255))
    paste_x = (max_size - width) // 2
    paste_y = (max_size - height) // 2
    square_image.paste(image, (paste_x, paste_y))  
    return square_image

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
model.eval()
processor = AutoProcessor.from_pretrained(model_path)

save_dir = "./examples"
image_path = "./examples/ori_image.png"
image = Image.open(image_path)
image = pad_to_square(image)
if image.size[0] < 224:
    image = image.resize((224, 224))
elif image.size[0] > 560:
    image = image.resize((560, 560))

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please convert the image to python code."},
            {"type": "image", "image": image},
        ],
    }
]

with torch.no_grad():
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    sucess, code = save_image_from_response(output_text, os.path.join(save_dir, "code_rec_image.png"))
    with open(os.path.join(save_dir, "code_rec_image.txt"), "w", encoding="utf-8") as f:
        f.write(code)
