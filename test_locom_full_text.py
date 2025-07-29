import json
import torch
#from modeling_llama_qk_retrieve import LlamaForCausalLM
from modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer

model_path = "/mnt/sda1/Llama-3.1-8B"  # 替换为你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
#model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, block_size = 32, topk = 1).cuda()
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
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
    prompt = full_history_prompt + "\nQuestion: " + question + "\nAnswer: "
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
    print("++",prompt)
    print("+++++++++++++++++++++++++++++")
    print("=============================")
    print("@@",answer)
    print("============================")
    
    # 保存结果
    results.append({
        "question": question,
        "standard answer": standard_answer,
        "answer": answer
    })
# 保存结果到文件
with open("full_text_ans.txt", "w") as f:
    json.dump(results, f, indent=4)

print("\n\n")
print("Results saved to full_text_ans.txt")