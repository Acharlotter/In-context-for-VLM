from huggingface_hub import snapshot_download

repo_id = "Qwen/Qwen2.5-VL-3B-Instruct"
local_dir = "./Qwen2.5-VL-3B-Instruct"

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 真实拷贝，适合集群/容器
    resume_download=True
)
