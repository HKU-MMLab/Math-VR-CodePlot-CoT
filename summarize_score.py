import json
import argparse
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="./examples/result.json", help="The path to save the evaluation report json file.")
    parser.add_argument("--data_path", type=str, default="./data/test-00000-of-00001.parquet", help="The path to the benchmark dataset.")
    args = parser.parse_args()
    return args

def summerize(result_dict, dataset):
    total_ac = 0
    total_ps = 0
    wrong_discount = 0.7
    total_num = 0

    for key in result_dict.keys():
        try:
            total_num += 1
            max_score = dataset[int(key)]['total_score']
            score = 1.0 if result_dict[key]['is_fully_correct'] else result_dict[key]['final_score'] / max_score * wrong_discount
            total_ps += score
            total_ac += (result_dict[key]['is_fully_correct'] is True)
        except Exception as e:
            print(f"Error processing key {key}: {e}")
    if total_num == 0:
        return 0.0, 0.0
    else:
        return total_ac / total_num * 100, total_ps / total_num * 100

args = parse_args()
dataset = load_dataset("parquet", data_files={"test": args.data_path})["test"]

with open(args.result_dir, 'r') as f:
    result_dict = json.load(f)

result_dict_text = {k: v for k, v in result_dict.items() if dataset[int(k)]['category'] == 'text'}
result_dict_mm = {k: v for k, v in result_dict.items() if dataset[int(k)]['category'] == 'multimodal'}

ac_text, ps_text = summerize(result_dict_text, dataset)
ac_mm, ps_mm = summerize(result_dict_mm, dataset)
ac_overall, ps_overall = summerize(result_dict, dataset)

print(f"Text: Accuracy Score = {ac_text:.2f}, Process Score = {ps_text:.2f}")
print(f"Multimodal: Accuracy Score = {ac_mm:.2f}, Process Score = {ps_mm:.2f}")
print(f"Overall: Accuracy Score = {ac_overall:.2f}, Process Score = {ps_overall:.2f}")
