import torch, os
import pandas as pd

def get_csv(is_causal, self_attn, model_name, mode):
    causal_flag = "_causal" if is_causal else ""
    attention_type_flag = 'self_attn_' if self_attn else 'cross_attn_'
    usage_csv_name = f"../results/{model_name}{causal_flag}_{attention_type_flag}mha_{mode}_usage.csv"
    return usage_csv_name

def benchmark(
    model_name,
    get_model,
    is_causal,
    n_repeat=5,
    batch_size = 2, 
    seed = 0,
    embed_dim = 32,
    num_heads = 2,
    self_attn = True,
    overwrite = False,
    max_len_power = 21, 
):
    torch.manual_seed(seed)

    model = get_model(model_name, d_model = embed_dim, n_heads = num_heads)

    def run_model(model, q, k, v, is_causal, self_attn = False):
        if 'tf' in model_name:
            if self_attn: 
                model(q, is_causal = is_causal)
            else:
                x = torch.concatenate([k, q], dim = 1)
                model(x, src_mask = k.shape[0], is_causal = is_causal)
        else:
            model(q, k, v, is_causal = is_causal, need_weights = False)

    seq_len = [2**i for i in range(4, max_len_power+1)]
    time_usage, memory_usage = {}, {}
    time_usage_csv_name = get_csv(is_causal, self_attn, model_name, 'time')
    memory_usage_csv_name = get_csv(is_causal, self_attn, model_name, 'memory')
    if os.path.exists(time_usage_csv_name) and os.path.exists(memory_usage_csv_name) and not overwrite:
        print(f"{time_usage_csv_name} and {memory_usage_csv_name} already exists, skipping")
        return
    
    for sl in seq_len:
        
        time_usage[sl], memory_usage[sl] = [], []

        if self_attn:
            q = torch.randn(batch_size*num_heads, sl, embed_dim, device = "cuda")
            k = q
            v = q
        else:
            q = torch.randn(batch_size*num_heads, 100, embed_dim, device = "cuda")
            k = torch.randn(batch_size*num_heads, sl, embed_dim, device = "cuda")
            v = torch.randn(batch_size*num_heads, sl, embed_dim, device = "cuda")

        # record cuda time
        for i in range(n_repeat):
            if i == 0: 
                # warmup
                run_model(model, q, k, v, is_causal, self_attn)
            else:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                torch.cuda.reset_peak_memory_stats(device='cuda')
                memory_before = torch.cuda.memory_allocated(device="cuda")
                start.record()
                run_model(model, q, k, v, is_causal, self_attn)
                end.record()
                torch.cuda.synchronize()
                memory_after = torch.cuda.max_memory_allocated(device="cuda")

                time_usage[sl].append(start.elapsed_time(end))
                memory_usage[sl].append(memory_after - memory_before)

    time_usage_df, memory_usage_df = pd.DataFrame(time_usage), pd.DataFrame(memory_usage)
    time_usage_df.to_csv(time_usage_csv_name)
    memory_usage_df.to_csv(memory_usage_csv_name)