import torch
from modeling_llama_qk_retrieve import LlamaForCausalLM
#from modeling_llama import LlamaForCausalLM
from transformers import AutoTokenizer

# ✅ 加载模型和 tokenizer
model_path = "/mnt/sda1/Meta-Llama-3-8B-Instruct"  # 替换为你的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, block_size = 16, topk = 2).cuda()
#model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).cuda()
model.eval()

# ✅ 仅预设用户输入内容（用户轮）

# 1: [1, 29871, 30406, 31229, 30383, 30275, 30356, 31030, 30415, 31615, 233, 159, 178, 30257, 30415, 30956, 30909, 232, 150, 173, 30755, 13, 31931, 30880, 30383, 232, 139, 155, 233, 156, 150, 232, 182, 179]

# 2: [1, 29871, 30406, 31229, 30383, 30275, 30356, 31030, 30415, 31615, 233, 159, 178, 30257, 30415, 231, 190, 131, 31882, 30594, 31974, 30886, 31071, 13, 31931, 30880, 30383, 232, 139, 155, 233, 156, 150, 232]

user_inputs = [
    "介绍一下Transformer",
    "介绍矩阵计算加速技术",
    "介绍两者的关系",
    "中国科学技术大学位于哪里",
    "介绍这座城市",
]

# ✅ 初始化对话历史
dialog_history = []
past_key_values = None

# ✅ 构造 prompt 的工具函数
def build_prompt(dialog):
    prompt = ""
    maintain_history = False
    if maintain_history:
        for role, text in dialog:
            prefix = "用户：" if role == "用户" else "助手："
            prompt += f"{prefix}{text}\n"
    else:
        cnt = 0
        for role, text in dialog:
            cnt = cnt+1
            if cnt >= len(dialog):
                prefix = "用户：" if role == "用户" else "助手："
                prompt += f"{prefix}{text}\n"
    prompt += "助手："  # 当前让助手继续回复
    return prompt


# ✅ 多轮对话生成测试
for round_idx, user_input in enumerate(user_inputs):
    # 添加当前用户输入
    dialog_history.append(("用户", user_input))
    
    # 构造 prompt（可选优化：仅追加用户输入 + KV cache）
    prompt = build_prompt(dialog_history)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

    # 生成助手回复（动态）
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            #temperature = 1,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # 提取生成内容
    generated_tokens = output.sequences[0][inputs["input_ids"].shape[-1]:]
    assistant_reply = tokenizer.decode(generated_tokens, skip_special_tokens=False)


    print("\n=======================================================================")
    print(f"🧠 第 {round_idx + 1} 轮")
    print("=======================================================================\n")
    print("---------------------------------------------------------------------------------------")
    print(f"👤 用户：{user_input}")
    print("---------------------------------------------------------------------------------------\n")
    print("---------------------------------------------------------------------------------------")
    print(f"prompt ：{inputs}")
    print("---------------------------------------------------------------------------------------\n")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"🤖 助手：{assistant_reply}")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")


    # 将助手回复动态加入对话历史（供下一轮构造 prompt）
    dialog_history.append(("助手", assistant_reply))
    
    
