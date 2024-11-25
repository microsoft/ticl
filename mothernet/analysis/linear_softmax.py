# %%
import torch, os, sys
import numpy as np
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pandas as pd
from fast_transformers.builders import TransformerEncoderBuilder
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from fast_transformers.masking import FullMask, LengthMask

root_dir = os.path.dirname(os.getcwd())
sys.path.append(root_dir)
import pdb
import torch.nn.functional as F
import torch.nn as nn
import math

# %%
n_layers = 2
num_heads = 1
embed_dim = 32
n_hid = 128
vocab_size = 10
batch_size = 1024
device = 'cuda'
n_batches = 10

# %%
torch.manual_seed(0)
# Create the builder for our transformers
builder = TransformerEncoderBuilder.from_kwargs(
    n_layers=n_layers,
    n_heads=num_heads,
    query_dimensions=embed_dim // num_heads,
    value_dimensions=embed_dim // num_heads,
    feed_forward_dimensions=n_hid,
    dropout=0.1
)

# [vocab_size, embed_dim]
embedding_layer = nn.Embedding(vocab_size, embed_dim).to(device)
# get embedding of all tokens
token_embeddings = embedding_layer.weight.data
kv_pairs = torch.randint(0, vocab_size//2, (batch_size, vocab_size), device=device)


# %%
def generate_associative_recall_batch(seq_len=10, batch_size=32, vocab_size=10, device='cuda'):
    """Generate a batch of associative recall training data with discrete tokens"""
    batch_indices = torch.arange(batch_size, device=device)
    
    # [batch, seq_len]
    keys = torch.randint(0, vocab_size//2, (batch_size, seq_len), device=device)

    # [batch, seq_len, embed_dim] 
    keys_embedded = token_embeddings[keys + vocab_size//2]
    # [batch, seq_len]
    values = torch.gather(kv_pairs, 1, keys) 
    # [batch, seq_len, embed_dim]
    values_embedded = token_embeddings[values]
    
    # Randomly select query indices
    query_indices = torch.randint(0, seq_len, (batch_size,), device=device)
    queries = keys_embedded[batch_indices, query_indices]  # [batch, embed_dim]
    
    # True targets are the values corresponding to the queries
    targets = values[batch_indices, query_indices]  # [batch]
    
    # Prepare input sequence: interleave keys and values, then append query
    input_seq = torch.zeros(batch_size, seq_len*2, embed_dim, device=device)
    input_seq[:,0::2] = keys_embedded   # Even indices for keys
    input_seq[:,1::2] = values_embedded # Odd indices for values]

    # input_seq = torch.zeros(batch_size, seq_len*2, device=device)
    # input_seq[:, 0::2] = keys
    # input_seq[:, 1::2] = values
    # pdb.set_trace()
    input_seq = torch.cat([input_seq, queries.unsqueeze(1)], dim=1)  # [batch, seq_len*2+1, embed_dim]
    # Add positional embeddings
    positions = torch.arange(input_seq.size(1), device=device).unsqueeze(0)  # [1, seq_len*2+1]
    positions = positions.expand(batch_size, -1)  # [batch, seq_len*2+1]
    
    # Simple sinusoidal positional embeddings
    pos_dim = embed_dim
    div_term = torch.exp(torch.arange(0, pos_dim, 2, device=device).float() * (-math.log(10000.0) / pos_dim))
    pe = torch.zeros(1, input_seq.size(1), pos_dim, device=device)
    pe[0, :, 0::2] = torch.sin(positions.float().unsqueeze(-1) * div_term)
    pe[0, :, 1::2] = torch.cos(positions.float().unsqueeze(-1) * div_term)
    
    # Add positional embeddings to input sequence
    input_seq = input_seq + pe
    return input_seq, targets


# %%

def train_model(model, seq_len, vocab_size=10, n_epochs=1000, batch_size=1024, device='cuda'):
    """Train model on associative recall classification task"""
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    
    for epoch in range(n_epochs):
        model.train()
        input_seq, targets = generate_associative_recall_batch(seq_len, batch_size, vocab_size, device)
        
        # Create attention mask
        # attention_mask = FullMask(seq_len*2 + 1)
        
        # Forward pass
        optimizer.zero_grad()

        output = model(input_seq)  # [batch, seq_len*2+1, vocab_size]
        predictions = output[:, -1]  # Take last token predictions [batch, vocab_size]
        
        # Compute loss
        loss = criterion(predictions, targets)
        # Compute accuracy
        pred_labels = predictions.argmax(dim=-1)
        acc = (pred_labels == targets).float().mean()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}, Acc: {acc.item():.6f}")
    
    return np.mean(losses[-10:])  # Return average of last 10 losses

# %%
def evaluate_model(model, seq_len, vocab_size=10, n_batches=10, batch_size=32, device='cuda'):
    """Evaluate model on associative recall classification task"""
    model.eval()
    total_acc = 0
    
    with torch.no_grad():
        for _ in range(n_batches):
            input_seq, targets = generate_associative_recall_batch(seq_len, batch_size, vocab_size, device)
            
            # attention_mask = FullMask(seq_len*2 + 1)
            
            output = model(input_seq)
            predictions = output[:, -1].argmax(dim=-1)  # [batch]
            
            acc = (predictions == targets).float().mean()
            total_acc += acc.item()
    
    return total_acc / n_batches

# %%
# Run experiments
seq_lengths = [20]
results = {'linear': [], 'softmax': []}

print("\nTraining and evaluating models...")
print("\nSequence Length | Linear Acc | Softmax Acc")
print("-" * 45)

for seq_len in seq_lengths:
    # Reset models
    builder.attention_type = "full" 
    softmax_model = nn.Sequential(builder.get(), nn.Linear(embed_dim, vocab_size//2)).to('cuda')
    builder.attention_type = "linear"
    linear_model = nn.Sequential(builder.get(), nn.Linear(embed_dim, vocab_size//2)).to('cuda')
    
    # Train models
    print(f"\nTraining on sequence length {seq_len}:")
    print("Linear attention model:")
    linear_train_loss = train_model(linear_model, seq_len, vocab_size, batch_size=batch_size)
    print("\nSoftmax attention model:")
    softmax_train_loss = train_model(softmax_model, seq_len, vocab_size, batch_size=batch_size)
    
    # Evaluate models
    linear_eval_acc = evaluate_model(linear_model, seq_len, vocab_size, n_batches=n_batches, batch_size=batch_size)
    softmax_eval_acc = evaluate_model(softmax_model, seq_len, vocab_size, n_batches=n_batches, batch_size=batch_size)
    
    results['linear'].append(linear_eval_acc)
    results['softmax'].append(softmax_eval_acc)
    
    print(f"\n{seq_len:14d} | {linear_eval_acc:.6f} | {softmax_eval_acc:.6f}")

# Create DataFrame with results
results_df = pd.DataFrame({
    'seq_length': seq_lengths,
    'linear_acc': results['linear'],
    'softmax_acc': results['softmax']
})



