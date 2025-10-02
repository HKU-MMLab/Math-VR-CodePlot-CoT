from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import os
from PIL import Image
import re
from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria
import textwrap

model_path = "./ckpts/CodePlot-CoT"

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_ids):
        self.stop_ids = stop_ids

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] in self.stop_ids

def save_image_from_response(response, image_filename):
    code = re.sub(r"plt\.show\s*\(\s*\)", f'plt.savefig("{image_filename}")', response)
    try:
        exec(code, {}, {})
        return True
    except:
        return False
    
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
stop_ids = processor.tokenizer.encode("<|code_end|>")[0]
eos_token_id = processor.tokenizer.eos_token_id
stopping_criteria = StoppingCriteriaList([StopOnTokens([stop_ids, eos_token_id])])
def solve(question, images, model, processor, save_dir):
    generate_image_cnt = 0
    text_analysis = ""
    finished = False 
    messages = [
        {
            "role": "user",
            "content": [],
        }
    ]
    for image in images:
        image = Image.open(image)
        image = pad_to_square(image)
        if image.size[0] < 224:
            image = image.resize((224, 224))
        elif image.size[0] > 560:
            image = image.resize((560, 560))
        messages[0]["content"].append({"type": "image", "image": image})
    messages[0]["content"].append({"type": "text", "text": question})


    with torch.no_grad():
        while (not finished and generate_image_cnt < 10):
            if (len(messages) > 0 and messages[-1]["role"] == "user"):
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            
            if generate_image_cnt != 0:
                text = text.removesuffix("<|im_end|>\n")

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            generated_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False, stopping_criteria=stopping_criteria)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            if generated_ids_trimmed[0][-1] == stop_ids:
                code = output_text.split("<|code_start|>")[1].split("<|code_end|>")[0]
                success = save_image_from_response(code, os.path.join(save_dir, f"image_{generate_image_cnt}.png"))
                if not success:
                    print("Failed to generate valid code for intermediate image.")
                    break

                with open(os.path.join(save_dir, f"code_{generate_image_cnt}.txt"), "w", encoding="utf-8") as f:
                    f.write(code)
                text_analysis += output_text.split("<|code_start|>")[0]
                text_analysis += f"\n![Image](image_{generate_image_cnt}.png)\n"
                image = Image.open(os.path.join(save_dir, f"image_{generate_image_cnt}.png"))
                image = pad_to_square(image).resize((448, 448))
                generate_image_cnt += 1

                if len(messages) == 1:
                    messages.append({
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": output_text},
                            {"type": "image", "image": image},
                        ],
                    })
                else:
                    messages[-1]["content"].append({"type": "text", "text": output_text})
                    messages[-1]["content"].append({"type": "image", "image": image})    
            else:
                text_analysis += output_text
                finished = True


    if not finished:
        print("Failed to generate complete analysis.")

    with open(os.path.join(save_dir, "analysis.md"), "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(text_analysis))


if __name__ == "__main__":
    # text only question example
    save_dir = "./results/text_question"
    os.makedirs(save_dir, exist_ok=True)
    question = """For an isosceles triangle $\\triangle ABC$ with a perimeter of 36 cm, if the altitude to its base $BC$ is 12 cm, what is the value of $\\cos B$? (     )

    **Options:**

    A. $\\dfrac{1}{2}$

    B. $\\dfrac{3}{2}$

    C. $\\dfrac{12}{13}$

    D. $\\dfrac{5}{13}$"""
    question = textwrap.dedent(question)
    images = []
    solve(question, images, model, processor, save_dir)

    # multimodal input question example
    save_dir = "./results/mm_question"
    os.makedirs(save_dir, exist_ok=True)
    question = """As shown, $MN$ is the diameter of circle $\\bigodot O$ and the radius of $\\bigodot O$ is 2. Point $A$ lies on $\\bigodot O$ and $\\angle AMN = 30^{\\circ}$. $B$ is the midpoint of arc $AN$. $P$ is a movable point on the diameter $MN$. Find the minimum value of $PA + PB$: __________ .""" 
    question = textwrap.dedent(question)
    images = ["examples/mm_question.jpg"]
    solve(question, images, model, processor, save_dir)