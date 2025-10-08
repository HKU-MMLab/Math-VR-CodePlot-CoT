import json
from openai import OpenAI
import base64
from tqdm import tqdm
import argparse
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_dir", type=str, default="./examples/answer.json", help="The path to the model answer json file.")
    parser.add_argument("--result_dir", type=str, default="./examples/result.json", help="The path to save the evaluation report json file.")
    parser.add_argument("--data_path", type=str, default="./data/test-00000-of-00001.parquet", help="The path to the benchmark dataset.")
    parser.add_argument("--api_key", type=str, default=None, help="The OpenAI API key.")
    args = parser.parse_args()
    return args

args = parse_args()

dataset = load_dataset("parquet", data_files={"test": args.data_path})["test"]

general_instruction_1 = """ \
You are an expert math teacher and grader. Your task is to evaluate a student's solution to a mathematical question and provide a score. 
You will be provided with the mathematical question (which may include multiple sub-questions), its ID, the student's solution, the correct answer, and the maximum possible score for the question below:
"""

general_instruction_2 = """ 
{{'id':{question_id}, 'student_solution':{student_solution}, 'correct_answer':{correct_answer}, 'max_score':{max_score}}}.

Please follow these steps precisely:

1. Initial Check for Correctness:
 - Thoroughly review the question and the student_solution to identify the student’s final answer.
 - Compare this final answer directly with the provided correct_answer. 
 - If the answers match exactly, award the full max_score.

2. Partial Credit Evaluation:
 - If the student's answer is not fully correct, evaluate its work for partial credit using the grading rubric: {{'scoring_points':{scoring_points}, 'point_values':{point_values}}}.
 - Go through each scoring_point, indicate if the student successfully completed that step.
 - Write down all the point_ids that the student earned and calculate the total score by summing the values of those points.

3. Provide your evaluation in a strict JSON format:
{{
  "id": "string",
  "question_solution_analysis": "string"
  "is_fully_correct": "boolean",
  "check_scoring_point":"string",
  "awarded_points": ["all" OR a list of earned point_ids like "p1", "p2"],
  "final_score": "number"
}}

Field Explanations:
 - "id": question id.
 - "question_solution_analysis": Analyze the question requirements and compare the student's answer against the correct_answer."
 - "is_fully_correct": True if the student's solution is fully correct, otherwise False.
 - "check_scoring_point": If fully correct, provide an empty string "". If not fully correct, explain where in the student_solution each scoring point is fully met or not met.
 - "awarded_points": If fully correct, this should be ["all"]. If partially correct, provide a list containing the fully met point_ids (e.g., ["p1", "p3"]). If no points met, provide an empty list [].
 - "final_score": the max_score if fully correct, or the sum of partial scores otherwise.

Provide your answer in ENGLISH!
"""

def extract_json(string):
    start = string.find('{')
    end = string.rfind('}') + 1
    json_part = string[start:end]
    try:
        json_data = json.loads(json_part)
    except json.JSONDecodeError:
        return None
    return json_data

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image

def content_list_creation(instruction_prepare_1, instruction_prepare_2, all_text,all_images):
    message_list = [
        {"type": "text", "text": instruction_prepare_1},
        {"type": "text", "text": all_text},
    ]
    encoded_images = []
    for img in all_images:
        tmp = encode_image(img)
        encoded_images.append(tmp)

    for encoded_image in encoded_images:
        message_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
        })
    message_list.append(
        {"type": "text", "text": instruction_prepare_2}
    )
    return message_list

results = {}
    
with open(args.answer_dir, 'r') as file:
    model_answer_dict = json.load(file)

client = OpenAI(
    api_key=args.api_key,
)

for question_id in tqdm(list(model_answer_dict.keys())):
    all_text = f"\n\n---\n\nquestion:\n{dataset[int(question_id)]['question']}"
    all_images = []
    correct_answer = {'total_answer':dataset[int(question_id)]['number_of_answers'], 'answer_summary':dataset[int(question_id)]['answers']}
    max_score = dataset[int(question_id)]['total_score']
    scoring_points = dataset[int(question_id)]['scoring_points']
    point_values = dataset[int(question_id)]['scores']
    student_solution = model_answer_dict[question_id]
    instruction_prepare_1 = general_instruction_1
    instruction_prepare_2 = general_instruction_2.format(
        question_id=question_id, 
        student_solution=student_solution, 
        correct_answer=correct_answer, 
        max_score=max_score, 
        scoring_points=scoring_points, 
        point_values=point_values
    )
    content_list = content_list_creation(instruction_prepare_1,instruction_prepare_2,all_text,all_images)
    for _ in range(5):  # Try up to 5 times to get a valid JSON response
        completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": content_list,
                }
            ]
        )
        answer = completion.choices[0].message.content
        data = extract_json(answer)
        if data is not None:
            results[question_id] = data
            break 
    if question_id not in results:
        print(f"Warning: {question_id} is not evaluated due to invalid json output by GPT4.1.")

    with open(args.result_dir, 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
