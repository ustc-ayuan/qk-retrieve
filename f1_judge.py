import os
import argparse
import json
import re
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 定义 F1 分数计算函数
def calculate_f1_score(gold_answer, generated_answer):
    """Calculate the F1 score between the gold answer and the generated answer."""
    # 去掉标点符号并转换为小写
    gold_answer = re.sub(r'[^\w\s]', '', gold_answer).lower()
    generated_answer = re.sub(r'[^\w\s]', '', generated_answer).lower()
    
    # 分词
    gold_tokens = set(gold_answer.split())
    generated_tokens = set(generated_answer.split())
    
    # 检查生成答案是否包含标准答案中的所有单词
    if gold_tokens.issubset(generated_tokens):
        return 1.0  # 如果包含所有单词，则认为完全正确
    
    # 如果不完全包含，则计算 F1 分数
    intersection = gold_tokens.intersection(generated_tokens)
    precision = len(intersection) / len(generated_tokens) if generated_tokens else 0
    recall = len(intersection) / len(gold_tokens) if gold_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    return f1

# 定义 BLEU 分数计算函数
def calculate_bleu_score(gold_answer, generated_answer):
    """Calculate the BLEU-1 score between the gold answer and the generated answer."""
    # 去掉标点符号并转换为小写
    gold_answer = re.sub(r'[^\w\s]', '', gold_answer).lower()
    generated_answer = re.sub(r'[^\w\s]', '', generated_answer).lower()
    
    # 分词
    gold_tokens = [gold_answer.split()]
    generated_tokens = generated_answer.split()
    
    # 使用平滑函数避免0分
    chencherry = SmoothingFunction()
    bleu_score = sentence_bleu(gold_tokens, generated_tokens, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)
    
    return bleu_score

def clean_answer(answer):
    """Clean the answer by removing content after '\nQuestion' and removing punctuation."""
    if '\nQuestion' in answer:
        answer = answer.split('\nQuestion')[0].strip()
    # 去掉标点符号
    answer = re.sub(r'[^\w\s]', '', answer)
    return answer

def process_file(file_path):
    """Process a single file and calculate F1 and BLEU-1 scores."""
    with open(file_path, "r") as f:
        data = json.load(f)

    total_f1 = 0
    total_bleu = 0
    total_questions = 0
    for item in data:
        gold_answer = str(item["standard answer"])
        generated_answer = clean_answer(item["answer"])  # Clean the answer

        # Calculate scores
        f1 = calculate_f1_score(gold_answer, generated_answer)
        bleu1 = calculate_bleu_score(gold_answer, generated_answer)
        
        item["f1_score"] = f1
        item["bleu1_score"] = bleu1
        item["answer"] = generated_answer  # Replace the original answer with the cleaned answer
        
        total_f1 += f1
        total_bleu += bleu1
        total_questions += 1

    # Calculate average scores
    average_f1 = total_f1 / total_questions if total_questions > 0 else 0
    average_bleu = total_bleu / total_questions if total_questions > 0 else 0

        
    # Save the judged data to a new file
    judged_file_name = f"Judged_{os.path.basename(file_path)}"
    judged_file_path = os.path.join(os.path.dirname(file_path), judged_file_name)
    with open(judged_file_path, "w") as f:
        json.dump(data, f, indent=4)

    return average_f1, average_bleu, judged_file_path

def main():
    """Main function to evaluate RAG results using F1 score."""
    parser = argparse.ArgumentParser(description="Evaluate RAG results using F1 score")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input file or folder containing result files.",
    )
    parser.add_argument(
        "--is_file",
        action="store_true",
        help="Indicate if the input is a single file. If not provided, the input is treated as a folder.",
    )
    args = parser.parse_args()

    input_path = args.input
    is_file = args.is_file

    if is_file:
        # Process a single file
        average_f1, average_bleu, judged_file_path = process_file(input_path)
        print(f"Average F1 Score: {average_f1:.4f}")
        print(f"Average BLEU-1 Score: {average_bleu:.4f}")
        print(f"Judged results saved to {judged_file_path}")
    else:
        # Process all files in the folder
        pattern = re.compile(r"output_block_(\d+)_topk_(\d+(\.\d+)?)\.txt")
        f1_accuracies = {}
        bleu_accuracies = {}

        for file_name in tqdm(os.listdir(input_path)):
            match = pattern.match(file_name)
            if match:
                block_size = int(match.group(1))
                topk = float(match.group(2))
                file_path = os.path.join(input_path, file_name)

                average_f1, average_bleu, _ = process_file(file_path)
                f1_accuracies[(block_size, topk)] = average_f1
                bleu_accuracies[(block_size, topk)] = average_bleu

        # Extract unique block sizes and topk values
        block_sizes = sorted(set(bs for bs, _ in f1_accuracies.keys()))
        topk_values = sorted(set(tk for _, tk in f1_accuracies.keys()))

        # Print the F1 score table
        print("\n--- F1 Scores ---")
        print("Block Size / Topk", end="\t")
        for topk in topk_values:
            print(f"Topk {topk}", end="\t")
        print()

        for block_size in block_sizes:
            print(f"Block {block_size}", end="\t")
            for topk in topk_values:
                average_f1 = f1_accuracies.get((block_size, topk), 0)
                print(f"{average_f1:.4f}", end="\t")
            print()

        # Print the BLEU-1 score table
        print("\n--- BLEU-1 Scores ---")
        print("Block Size / Topk", end="\t")
        for topk in topk_values:
            print(f"Topk {topk}", end="\t")
        print()

        for block_size in block_sizes:
            print(f"Block {block_size}", end="\t")
            for topk in topk_values:
                average_bleu = bleu_accuracies.get((block_size, topk), 0)
                print(f"{average_bleu:.4f}", end="\t")
            print()

if __name__ == "__main__":
    main()
