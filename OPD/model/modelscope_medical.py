from modelscope import snapshot_download

# 下载 70B 教师模型，保存到 OPD/model/DeepSeek/DeepSeek-R1-Distill-Llama-70B
print("Downloading DeepSeek 70B...")
snapshot_download(
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 
    local_dir='./OPD/model/DeepSeek/DeepSeek-R1-Distill-Qwen-7B',
    # 忽略掉传统的 pytorch 权重文件，只下载更高效的 safetensors 格式
    ignore_file_pattern=['*.bin', '*.pt', '*.pth'] 
)

# 下载 4B 学生模型，保存到 OPD/model/Qwen/Qwen3.5-4B
print("Downloading Qwen 4B...")
snapshot_download(
    'qwen/Qwen3-0.6B', 
    local_dir='./OPD/model/Qwen/Qwen3-0.6B',
    ignore_file_pattern=['*.bin', '*.pt', '*.pth']
)