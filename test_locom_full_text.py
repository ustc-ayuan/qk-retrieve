import json
import torch
import argparse
from modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Run LlamaForCausalLM with full text.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the results.")
    args = parser.parse_args()

    model_path = args.model_path
    output_path = args.output_path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    data_path = "./locomo10.json"
    with open(data_path, "r") as f:
        all_data = json.load(f)

    results = []  # 用于存储所有对话的结果
    
    for data in all_data:
        print("Starting new conversation...")
        # 每次对话重新加载模型
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
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

        system_prompt = """ You will be given some conversations with the Timestamp, you need to remember it
        and answer the question about these conversations. When it comes to time-related questions, provide
        specific dates and time rather than vague answers like "yesterday", "last month", "this week" and so on.\n
        """
        full_history_prompt = system_prompt
        for session in sessions:
            temp_prompt = full_history_prompt + "\nTimestamp: " + session['timestamp'] + "\nConversation: " + session['conversation']
            full_history_prompt = temp_prompt

        qas = data["qa"]
        for qa in qas:
            if "adversarial_answer" in qa:
                continue

            question = qa["question"]
            standard_answer = qa["answer"]
            
            prompt = full_history_prompt + "\nQuestion: " + question + "\n answer the question in short. \nAnswer: "
            
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

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n\n")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
