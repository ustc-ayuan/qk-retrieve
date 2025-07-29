import json
import torch
import argparse
from modeling_llama_qk_retrieve import LlamaForCausalLM
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Run LlamaForCausalLM with different parameters.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--block_size", type=int, required=True, help="Block size for the model.")
    parser.add_argument("--topk", type=int, required=True, help="Top-k value for the model.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the results.")
    args = parser.parse_args()

    model_path = args.model_path
    block_size = args.block_size
    topk = args.topk
    output_path = args.output_path

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, block_size=block_size, topk=topk).cuda()
    model.eval()

    data_path = "./locomo10.json"
    with open(data_path, "r") as f:
        data = json.load(f)[0]

    # 提取 conversation 数据
    conversation = data["conversation"]
    speaker_a = conversation["speaker_a"]
    speaker_b = conversation["speaker_b"]

    # 初始化 session 列表
    sessions = []

    # 遍历 conversation 中的每个 session
    for key in conversation.keys():
        if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
            continue

        # 提取 session 的时间戳和聊天记录
        date_time_key = key + "_date_time"
        timestamp = conversation[date_time_key]
        chats = conversation[key]

        # 初始化 messages 列表
        messages = []

        # 遍历每个聊天记录
        for chat in chats:
            messages.append(f"{chat['speaker']}: {chat['text']}")

        # 将 memory_prompt 和聊天记录拼接成一个完整的 prompt
        context = "\n".join(messages)

        # 将拼接后的 prompt 添加到 sessions 列表中
        sessions.append({
            "timestamp": timestamp,
            "conversation": context,
        })

    # 打印结果
    full_history_prompt = ""
    cnt = 0
    for session in sessions:
        cnt = cnt + 1
        if cnt > 13:
            break
        full_history_prompt = full_history_prompt + "\nTimestamp: " + session['timestamp'] + "\nConversation: " + session['conversation']

    qas = data["qa"]
    cnt = 0
    results = []  # 用于存储结果
    for qa in qas:
        question = qa["question"]
        standard_answer = qa["answer"]  # 假设每个问题都有一个标准答案
        cnt = cnt + 1
        if cnt > 150:
            break
        if cnt == 1:
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

        # 打印结果
        print("+++++++++++++++++++++++++++++")
        print("++", prompt)
        print("+++++++++++++++++++++++++++++")
        print("=============================")
        print("@@", answer)
        print("============================")

        # 保存结果
        results.append({
            "question": question,
            "standard answer": standard_answer,
            "answer": answer
        })

    # 保存结果到文件
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print("\n\n")
    print("Results saved to", output_path)

if __name__ == "__main__":
    main()