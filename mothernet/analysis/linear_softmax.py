import torch, os, sys, pdb
import numpy as np
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pandas as pd
from fast_transformers.builders import TransformerEncoderBuilder
import torch.nn.functional as F
import torch.nn as nn

# Parameters
n_layers = 2
num_heads = 1
embed_dim = 32
n_hid = 128
vocab_size = 10
device = 'cuda'
seq_lengths = [2]
batch_size = 1024
kv_pairs = torch.randint(0, vocab_size // 2, (batch_size, vocab_size), device=device)

# Transformer builder
torch.manual_seed(0)
builder = TransformerEncoderBuilder.from_kwargs(
    n_layers=n_layers,
    n_heads=num_heads,
    query_dimensions=embed_dim // num_heads,
    value_dimensions=embed_dim // num_heads,
    feed_forward_dimensions=n_hid,
    dropout=0.5
)

# Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device='cuda'):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # Shape [1, max_len, d_model]
        self.register_buffer('pe', pe)
        

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Model Class
class AssociativeRecallModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, builder):
        super(AssociativeRecallModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, device=device)
        self.transformer_encoder = builder.get()
        self.fc_out = nn.Linear(embed_dim, vocab_size // 2)

    
    def forward(self, input_seq):
        embedded = self.embedding(input_seq)  # Shape [batch_size, seq_len, embed_dim]
        embedded = self.pos_encoder(embedded)
        output = self.transformer_encoder(embedded)  # Shape [batch_size, seq_len, embed_dim]
        output = self.fc_out(output[:, -1, :])  # Use the output corresponding to the query
        return output

# Data Generation Function
def generate_associative_recall_batch(seq_len=10, batch_size=32, vocab_size=10, device='cuda'):
    """Generate a batch of associative recall training data with discrete tokens."""
    batch_indices = torch.arange(batch_size, device=device)
    
    keys = torch.randint(0, vocab_size // 2, (batch_size, seq_len), device=device)
    values = torch.gather(kv_pairs, 1, keys)
    
    # Randomly select query indices
    query_positions = torch.randint(0, seq_len, (batch_size,), device=device)
    queries = keys[batch_indices, query_positions] + vocab_size // 2 # Queries are keys
    
    # True targets are the values corresponding to the queries
    targets = values[batch_indices, query_positions]  # Shape [batch_size]
    
    # Prepare input sequence: interleave keys and values, then append query
    input_seq = torch.zeros(batch_size, seq_len * 2, dtype=torch.long, device=device)
    input_seq[:, 0::2] = keys + vocab_size // 2 # Even indices for keys
    input_seq[:, 1::2] = values  # Shift values to a different index range
    
    input_seq = torch.cat([input_seq, queries.unsqueeze(1)], dim=1)  # Shape [batch_size, seq_len * 2 + 1]

    return input_seq, targets

# Training Function
def train_model(model, seq_len, vocab_size=10, n_epochs=500, batch_size=1024, device='cuda'):
    """Train model on associative recall classification task."""
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)

    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    
    for epoch in range(n_epochs):
        model.train()
        input_seq, targets = generate_associative_recall_batch(seq_len, batch_size, vocab_size, device)
        
    
        predictions = model(input_seq)  # Shape [batch_size, vocab_size // 2]
        
        loss = criterion(predictions, targets)
        loss.backward()


        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 20 == 0:
            pred_labels = predictions.argmax(dim=-1)
            acc = (pred_labels == targets).float().mean()
            # print(model.transformer_encoder.layers[0].attention.query_projection.bias[0])
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.6f}, Acc: {acc.item():.6f}")
    
    return np.mean(losses[-10:])  # Return average of last 10 losses

# Evaluation Function
def evaluate_model(model, seq_len, vocab_size=10, n_batches=10, batch_size=32, device='cuda'):
    """Evaluate model on associative recall classification task."""
    model.eval()
    total_acc = 0
    
    with torch.no_grad():
        for _ in range(n_batches):
            input_seq, targets = generate_associative_recall_batch(seq_len, batch_size, vocab_size, device)
            predictions = model(input_seq)
            pred_labels = predictions.argmax(dim=-1)
            acc = (pred_labels == targets).float().mean()
            total_acc += acc.item()
    
    return total_acc / n_batches

# Run experiments
results = {'linear': [], 'softmax': []}

print("\nTraining and evaluating models...")
print("\nSequence Length | Linear Acc | Softmax Acc")
print("-" * 45)

for seq_len in seq_lengths:
    # Reset models
    builder.attention_type = "full"
    softmax_model = AssociativeRecallModel(vocab_size, embed_dim, builder).to(device)
    
    builder.attention_type = "linear"
    linear_model = AssociativeRecallModel(vocab_size, embed_dim, builder).to(device)


    
    # Train models
    print(f"\nTraining on sequence length {seq_len}:")
    print("\nSoftmax attention model:")
    train_model(softmax_model, seq_len, vocab_size, batch_size=batch_size)

    print("Linear attention model:")
    train_model(linear_model, seq_len, vocab_size, batch_size=batch_size)

    
    # Evaluate models
    softmax_eval_acc = evaluate_model(softmax_model, seq_len, vocab_size, n_batches=10, batch_size=batch_size)
    linear_eval_acc = evaluate_model(linear_model, seq_len, vocab_size, n_batches=10, batch_size=batch_size)
    
    results['linear'].append(linear_eval_acc)
    results['softmax'].append(softmax_eval_acc)
    
    print(f"\n{seq_len:14d} | {linear_eval_acc:.6f} | {softmax_eval_acc:.6f}")

# Create DataFrame with results
results_df = pd.DataFrame({
    'seq_length': seq_lengths,
    'linear_acc': results['linear'],
    'softmax_acc': results['softmax']
})
print(results_df)
