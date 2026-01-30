import torch
from qwen_tts import Qwen3TTSModel
from .config import PROJECT_CONFIG

def get_torch_dtype():
    dtype_str = PROJECT_CONFIG.get('torch_dtype', 'bfloat16')
    if dtype_str == 'bfloat16':
        return torch.bfloat16
    elif dtype_str == 'float16':
        return torch.float16
    else:
        return torch.float32

def load_voice_design_model():
    print(f"Loading Voice Design model: {PROJECT_CONFIG['voice_design_model']}...")
    
    attn_implementation = PROJECT_CONFIG.get('attn_implementation', 'sdpa')
    if PROJECT_CONFIG.get('device') == 'cpu':
        attn_implementation = 'eager'

    model = Qwen3TTSModel.from_pretrained(
        PROJECT_CONFIG['voice_design_model'],
        device_map=PROJECT_CONFIG['device'],
        dtype=get_torch_dtype(),
        attn_implementation=attn_implementation,
    )
    return model

def load_voice_clone_model():
    print(f"Loading Voice Clone model: {PROJECT_CONFIG['voice_clone_model']}...")

    attn_implementation = PROJECT_CONFIG.get('attn_implementation', 'sdpa')
    if PROJECT_CONFIG.get('device') == 'cpu':
        attn_implementation = 'eager'

    model = Qwen3TTSModel.from_pretrained(
        PROJECT_CONFIG['voice_clone_model'],
        device_map=PROJECT_CONFIG['device'],
        dtype=get_torch_dtype(),
        attn_implementation=attn_implementation,
    )
    return model
