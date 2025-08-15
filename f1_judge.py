import os
import argparse
import json
import re
import numpy as np
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from sentence_transformers import SentenceTransformer
import torch
from bert_score import score as bert_scorer

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
    #if gold_tokens.issubset(generated_tokens):
    #    return 1.0  # 如果包含所有单词，则认为完全正确
    
    # 如果不完全包含，则计算 F1 分数
    intersection = gold_tokens.intersection(generated_tokens)
    precision = len(intersection) / len(generated_tokens) if generated_tokens else 0
    recall = len(intersection) / len(gold_tokens) if gold_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    return f1

# 定义 BLEU-1 分数计算函数
def calculate_bleu1_score(gold_answer, generated_answer):
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

# --- New metric functions ---

def calculate_rouge_l_score(gold_answer, generated_answer):
    """Calculate the ROUGE-L (RL) score."""
    if not generated_answer.strip() or not gold_answer.strip():
        return 0.0
    rouge = Rouge()
    try:
        scores = rouge.get_scores(generated_answer, gold_answer, avg=True)
        return scores['rouge-l']['f']
    except (ValueError, KeyError):
        return 0.0

def calculate_bleu2_score(gold_answer, generated_answer):
    """Calculate the BLEU-2 (B2) score."""
    gold_answer = re.sub(r'[^\w\s]', '', gold_answer).lower()
    generated_answer = re.sub(r'[^\w\s]', '', generated_answer).lower()
    gold_tokens = [gold_answer.split()]
    generated_tokens = generated_answer.split()
    chencherry = SmoothingFunction()
    return sentence_bleu(gold_tokens, generated_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)

# --- End of new metric functions ---

def clean_answer(answer):
    """Clean the answer by removing content after '\nQuestion' and removing punctuation."""
    if '\nQuestion' in answer:
        answer = answer.split('\nQuestion')[0].strip()
    # 去掉标点符号
    answer = re.sub(r'[^\w\s]', '', answer)
    return answer

def process_file(file_path, embedding_model=None, bert_model_path='microsoft/deberta-xlarge-mnli'):
    """Process a single file and calculate all specified scores."""
    with open(file_path, "r") as f:
        data = json.load(f)

    if not data:
        return (0.0,) * 6 + (None,)

    golds = [str(item["standard answer"]) for item in data]
    gens = [clean_answer(item["answer"]) for item in data]

    # --- Calculate all scores ---
    f1_scores = [calculate_f1_score(g, gen) for g, gen in zip(golds, gens)]
    bleu1_scores = [calculate_bleu1_score(g, gen) for g, gen in zip(golds, gens)]
    rouge_l_scores = [calculate_rouge_l_score(g, gen) for g, gen in zip(golds, gens)]
    bleu2_scores = [calculate_bleu2_score(g, gen) for g, gen in zip(golds, gens)]
    
    # Batch calculation for BERTScore
    bert_scores = []
    if gens and golds:
        # Using a default model, user can specify another model if needed
        _, _, f1s = bert_scorer(gens, golds, lang="en", verbose=False, model_type=bert_model_path)
        bert_scores = f1s.tolist()

    # Batch calculation for Cosine Similarity
    sim_scores = []
    if embedding_model:
        gold_embeddings = embedding_model.encode(golds, convert_to_tensor=True, show_progress_bar=False)
        gen_embeddings = embedding_model.encode(gens, convert_to_tensor=True, show_progress_bar=False)
        cosine_sim = torch.nn.functional.cosine_similarity(gen_embeddings, gold_embeddings)
        sim_scores = cosine_sim.tolist()

    # --- Update data with scores and calculate averages ---
    num_questions = len(data)
    for i, item in enumerate(data):
        item["answer"] = gens[i]
        item["f1_score"] = f1_scores[i]
        item["bleu1_score"] = bleu1_scores[i]
        item["rouge_l_score"] = rouge_l_scores[i]
        item["bleu2_score"] = bleu2_scores[i]
        if bert_scores:
            item["bert_score_f1"] = bert_scores[i]
        if sim_scores:
            item["cosine_similarity"] = sim_scores[i]

    # --- Calculate average scores ---
    avg_f1 = sum(f1_scores) / num_questions
    avg_bleu1 = sum(bleu1_scores) / num_questions
    avg_rouge_l = sum(rouge_l_scores) / num_questions
    avg_bleu2 = sum(bleu2_scores) / num_questions
    avg_bert_score = sum(bert_scores) / num_questions if bert_scores else 0.0
    avg_sim = sum(sim_scores) / num_questions if sim_scores else 0.0

    # --- Save judged file ---
    judged_file_name = f"Judged_{os.path.basename(file_path)}"
    judged_file_path = os.path.join(os.path.dirname(file_path), judged_file_name)
    with open(judged_file_path, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return avg_f1, avg_bleu1, avg_rouge_l, avg_bleu2, avg_sim, avg_bert_score, judged_file_path

def main():
    """Main function to evaluate RAG results using multiple metrics."""
    parser = argparse.ArgumentParser(description="Evaluate RAG results using multiple metrics.")
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the input file or folder containing result files.",
    )
    parser.add_argument(
        "--is_file", action="store_true",
        help="Indicate if the input is a single file. If not provided, the input is treated as a folder.",
    )
    parser.add_argument(
        "--bert_model_path", type=str, default='microsoft/deberta-xlarge-mnli',
        help="Path to the local BERT model or huggingface model name for scoring."
    )
    args = parser.parse_args()



    # --- Load embedding model ---
    embedding_model_path = "/mnt/sda1/embedding_models/bge-m3"
    embedding_model = None
    if os.path.exists(embedding_model_path):
        print(f"Loading embedding model from {embedding_model_path}...")
        embedding_model = SentenceTransformer(embedding_model_path)
        print("Embedding model loaded.")
    else:
        print(f"Warning: Embedding model not found at {embedding_model_path}. Cosine similarity (Sim) will not be calculated.")

    if args.is_file:
        # Process a single file
        scores = process_file(args.input, embedding_model, args.bert_model_path)
        (avg_f1, avg_b1, avg_rl, avg_b2, avg_sim, avg_bert, judged_file_path) = scores
        print(f"\n--- Average Scores for {os.path.basename(args.input)} ---")
        print(f"F1 Score: {avg_f1:.4f}")
        print(f"BLEU-1 (B1): {avg_b1:.4f}")
        print(f"ROUGE-L (RL): {avg_rl:.4f}")
        print(f"BLEU-2 (B2): {avg_b2:.4f}")
        print(f"BERTScore F1: {avg_bert:.4f}")
        if embedding_model:
            print(f"Cosine Similarity (Sim): {avg_sim:.4f}")
        print(f"\nJudged results saved to {judged_file_path}")
    else:
        # Process all files in the folder
        pattern = re.compile(r"output_block_(\d+)_topk_(\d+(\.\d+)?)\.txt")
        accuracies = {
            "F1": {}, "B1": {}, "RL": {}, "B2": {}, "Sim": {}, "BERT": {}
        }
        files_to_process = [f for f in os.listdir(args.input) if pattern.match(f)]
        
        for file_name in tqdm(files_to_process, desc="Processing files"):
            match = pattern.match(file_name)
            block_size = int(match.group(1))
            topk = float(match.group(2))
            file_path = os.path.join(args.input, file_name)
            scores = process_file(file_path, embedding_model, args.bert_model_path)
            
            accuracies["F1"][(block_size, topk)] = scores[0]
            accuracies["B1"][(block_size, topk)] = scores[1]
            accuracies["RL"][(block_size, topk)] = scores[2]
            accuracies["B2"][(block_size, topk)] = scores[3]
            accuracies["Sim"][(block_size, topk)] = scores[4]
            accuracies["BERT"][(block_size, topk)] = scores[5]

        if not accuracies["F1"]:
            print("No matching files found to process in the specified folder.")
            return

        all_keys = accuracies["F1"].keys()
        block_sizes = sorted(set(bs for bs, _ in all_keys))
        topk_values = sorted(set(tk for _, tk in all_keys))

        # New printing logic for LaTeX format, incorporating user feedback
        print(r"\textbf{Threshold} & \textbf{F1 Score} & \textbf{BLEU-1} & \textbf{BLEU-2} & \textbf{ROUGE-L} & \textbf{BERTScore} & \textbf{Aver. Score}\\")
        
        for block_size in block_sizes:
            # Check if there is any data for this block size to avoid printing empty sections
            if not any(bs == block_size for bs, _ in all_keys):
                continue

            print(r"\midrule")
            # The header has 7 columns, so multicolumn should span 7 columns.
            print(f"\\multicolumn{{7}}{{c}}{{\\textbf{{Block {block_size}}}}} \\\\")
            print(r"\midrule")
            
            # Get the topk values that are actually present for this block_size
            # to ensure we don't print rows for non-existent combinations.
            block_specific_topks = sorted([tk for bs, tk in all_keys if bs == block_size])

            for topk in block_specific_topks:
                f1_score = accuracies["F1"].get((block_size, topk), 0) * 100
                bleu1_score = accuracies["B1"].get((block_size, topk), 0) * 100
                bleu2_score = accuracies["B2"].get((block_size, topk), 0) * 100
                rouge_l_score = accuracies["RL"].get((block_size, topk), 0) * 100
                bert_score = accuracies["BERT"].get((block_size, topk), 0) * 100
                
                avg_score = (f1_score + bleu1_score + bleu2_score + rouge_l_score + bert_score) / 5
                
                print(f"{topk} & {f1_score:.2f} & {bleu1_score:.2f} & {bleu2_score:.2f} & {rouge_l_score:.2f} & {bert_score:.2f} & {avg_score:.2f} \\\\")

if __name__ == "__main__":
    main()
