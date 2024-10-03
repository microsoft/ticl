import torch
from torch.nn import Module, Linear, LayerNorm, Dropout
import torch.nn.functional as F

from fast_transformers.events import EventDispatcher
from fast_transformers.masking import FullMask, LengthMask
from fast_transformers.attention import AttentionLayer
from fast_transformers.builders.attention_builders import AttentionBuilder
from fast_transformers.transformers import TransformerEncoder
from fast_transformers.builders.transformer_builders import BaseTransformerEncoderBuilder

class LinearAttentionTransformerEncoderLayer(Module):
    """Self attention and feed forward network with skip connections.

    This transformer encoder layer implements the same encoder layer as
    PyTorch but is a bit more open for extension by receiving the attention
    implementation as a constructor argument.

    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(LinearAttentionTransformerEncoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = getattr(F, activation)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, attn_mask = None, length_mask=None):
        """Apply the transformer encoder to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """

        # Normalize the masks
        batch_size, num_samples, d_model = x.shape

        if isinstance(attn_mask, int):
            # tabpfn
            single_eval_position = attn_mask
            train_samples = x[:,:single_eval_position,:]
            test_samples = x[:,single_eval_position:,:]

            num_train_samples = train_samples.shape[1]
            num_test_samples = test_samples.shape[1]

            attn_mask = None

            # Run self attention and add it to the input
            # the training samples are only attend to themselves
            attn_left = self.attention(
                train_samples, 
                train_samples, 
                train_samples, 
                attn_mask=None, 
                query_lengths=None,
                key_lengths=None,
            )

            # the testing samples attend to training samples
            attn_right = self.attention(
                test_samples,
                train_samples,
                train_samples,
                attn_mask=None, 
                query_lengths=None,
                key_lengths=None,
            )

            attn_output = torch.cat([attn_left, attn_right], dim=1)
        else:
            # mothernet
            # attn_mask = FullMask(num_samples, device=x.device)
            length_mask = length_mask or \
            LengthMask(x.new_full((batch_size,), num_samples, dtype=torch.int64))
            attn_output = self.attention(
                x, x, x,
                attn_mask=attn_mask,
                query_lengths=length_mask,
                key_lengths=length_mask
            )

        x = x + self.dropout(attn_output)

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        output = self.norm2(x+y)

        return output
      
class TransformerEncoderBuilder(BaseTransformerEncoderBuilder):
    """Build a batch transformer encoder for training or processing of
    sequences all elements at a time.

    Example usage:

        builder = TransformerEncoderBuilder()
        builder.n_layers = 12
        builder.n_heads = 8
        builder.feed_forward_dimensions = 1024
        builder.query_dimensions = 64
        builder.value_dimensions = 64
        builder.dropout = 0.1
        builder.attention_dropout = 0.1
        builder.attention_type = "linear"
        transformer = builder.get()
    """
    def _get_attention_builder(self):
        """Return an instance of the appropriate attention builder."""
        return AttentionBuilder()

    def _get_attention_layer_class(self):
        """Return the class for the layer that projects queries keys and
        values."""
        return AttentionLayer

    def _get_encoder_class(self):
        """Return the class for the transformer encoder."""
        return TransformerEncoder

    def _get_encoder_layer_class(self):
        """Return the class for the transformer encoder layer."""
        return LinearAttentionTransformerEncoderLayer
    