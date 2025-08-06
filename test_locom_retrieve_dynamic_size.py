import json
import torch
import argparse
from modeling_llama_qk_retrieve import LlamaForCausalLM
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Run LlamaForCausalLM with different parameters.")
    parser.add_argument("--topk_threshold", type=float, required=True, help="Top-k threshold for the model.")
    args = parser.parse_args()

    model_path = "/mnt/sda1/Llama-3.1-8B"
    topk_threshold = args.topk_threshold
    output_path = "./dynamic_block_size_ans.log"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    data_path = "./locomo10.json"
    with open(data_path, "r") as f:
        all_data = json.load(f)

    results = []  # 用于存储所有对话的结果
    
    for data in all_data:
        print("Starting new conversation...")
        # 每次对话重新加载模型
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, topk_threshold=topk_threshold).cuda()
        model.eval()

        # 提取 conversation 数据
        conversation = data["conversation"]

        # 初始化 session 列表
        sessions = []
        for key in sorted(conversation.keys()):
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue
            
            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]
            messages = [f"{chat['speaker']}: {chat['text']}" for chat in chats]
            context = "\n".join(messages)
            sessions.append({
                "timestamp": timestamp,
                "conversation": context,
            })

        # 保证 history_prompt 的 tokenizer 长度小于 7500
        full_history_prompt = ""
        for session in sessions:
            temp_prompt = full_history_prompt + "\nTimestamp: " + session['timestamp'] + "\nConversation: " + session['conversation']
            if len(tokenizer.encode(temp_prompt)) > 7500:
                break
            full_history_prompt = temp_prompt

        qas = data["qa"]
        for i, qa in enumerate(qas):
            if "adversarial_answer" in qa:
                continue

            question = qa["question"]
            standard_answer = qa["answer"]
            
            if i == 0:
                prompt = full_history_prompt + "\nQuestion: " + question + "\nAnswer: "
            else:
                prompt = "\nQuestion: " + question + "\nAnswer: "
            
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            generated_tokens = output.sequences[0][inputs["input_ids"].shape[-1]:]
            answer = tokenizer.decode(generated_tokens, skip_special_tokens=False)

            print("+++++++++++++++++++++++++++++")
            print("++", prompt)
            print("+++++++++++++++++++++++++++++")
            print("=============================")
            print("@@", answer)
            print("============================")

            results.append({
                "question": question,
                "standard answer": standard_answer,
                "answer": answer
            })

            # Save results incrementally to prevent data loss from OOM
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)

    # Final save is redundant if incremental saving is done, but kept for clarity
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n\n")
    print("Results saved to", output_path)

if __name__ == "__main__":
    main()
