import torch
import torch.nn as nn

from mothernet.models.layer import BiAttentionEncoderLayer
from mothernet.models.decoders import FactorizedAdditiveModelDecoder, AdditiveModelDecoder, SummaryLayer
from mothernet.models.encoders import BinEmbeddingEncoder, Linear, OneHotAndLinear, get_fourier_features
from mothernet.models.utils import bin_data

from mothernet.utils import SeqBN, get_init_method


class BiAttentionMotherNetAdditive(nn.Module):
    def __init__(self, *, n_features, n_out, emsize, nhead, nhid_factor, nlayers, dropout=0.0, y_encoder_layer=None,
                 input_normalization=False, init_method=None, pre_norm=False,
                 activation='gelu', recompute_attn=False, categorical_embedding=False,
                 all_layers_same_init=False, efficient_eval_masking=True, decoder_embed_dim=2048, low_rank_weights=None, weight_embedding_rank=None,
                 decoder_hidden_layers=1, decoder_hidden_size=None, n_bins=64, nan_bin=False, input_bin_embedding=False,
                 bin_embedding_rank=16, output_rank=16, factorized_output=False, y_encoder=None,
                 predicted_hidden_layer_size=None, predicted_hidden_layers=None,
                 decoder_type=None, input_layer_norm=False, shape_attention=False, tabpfn_zero_weights=True, shape_attention_heads=1, n_shape_functions=32,
                 shape_init="constant", decoder_activation='relu', fourier_features=0, marginal_residual=False):
        super().__init__()
        nhid = emsize * nhid_factor
        self.y_encoder = y_encoder_layer
        if categorical_embedding:
            self.categorical_embedding = OneHotAndLinear(num_classes=2, emsize=emsize)
        self.low_rank_weights = low_rank_weights  # ignored for now
        self.weight_embedding_rank = weight_embedding_rank  # ignored for now
        self.fourier_features = fourier_features

        def encoder_layer_creator(): return BiAttentionEncoderLayer(emsize, nhead, nhid, dropout, activation=activation,
                                                                    pre_norm=pre_norm, recompute_attn=recompute_attn)
        self.layers = nn.ModuleList([encoder_layer_creator() for _ in range(nlayers)])
        self.emsize = emsize

        if input_bin_embedding == "linear":
            self.encoder = BinEmbeddingEncoder(num_features=n_features, emsize=emsize, n_bins=n_bins, rank=bin_embedding_rank, nonlinear=False,
                                               decoder_activation=decoder_activation)
        elif input_bin_embedding in ["True", "nonlinear"] or isinstance(input_bin_embedding, bool) and input_bin_embedding:
            self.encoder = BinEmbeddingEncoder(num_features=n_features, emsize=emsize, n_bins=n_bins, rank=bin_embedding_rank, nonlinear=True,
                                               decoder_activation=decoder_activation)
        elif input_bin_embedding in ["none", "False"] or isinstance(input_bin_embedding, bool) and not input_bin_embedding:
            self.encoder = Linear(num_features=n_bins + (fourier_features + 1 if fourier_features else 0), emsize=emsize, replace_nan_by_zero=True)
        else:
            raise ValueError(f"Unknown input_bin_embedding: {input_bin_embedding}")

        self.input_ln = SeqBN(emsize) if input_normalization else None
        self.init_method = init_method
        self.efficient_eval_masking = efficient_eval_masking
        self.n_bins = n_bins
        self.nan_bin = nan_bin
        self.n_out = n_out
        self.nhid = nhid
        self.input_bin_embedding = input_bin_embedding
        self.output_rank = output_rank
        self.factorized_output = factorized_output
        self.decoder_type = decoder_type
        self.input_layer_norm = input_layer_norm
        self.shape_attention = shape_attention
        self.tabpfn_zero_weights = tabpfn_zero_weights
        self.decoder_activation = decoder_activation
        self.marginal_residual = marginal_residual
        if marginal_residual:
            self.marginal_residual_layer = nn.Linear(n_bins, n_bins, bias=False)
            self.marginal_residual_layer.weight.data = torch.eye(n_bins)
            self.class_average_layer = SummaryLayer(decoder_type="class_average", emsize=n_bins, n_out=n_out)

        if factorized_output:
            self.decoder = FactorizedAdditiveModelDecoder(n_features=n_features, n_bins=n_bins, emsize=emsize, hidden_size=decoder_hidden_size, n_out=n_out,
                                                          embed_dim=decoder_embed_dim, decoder_type=decoder_type,
                                                          decoder_hidden_layers=decoder_hidden_layers, nhead=nhead, rank=output_rank,
                                                          shape_attention=shape_attention, shape_attention_heads=shape_attention_heads,
                                                          n_shape_functions=n_shape_functions, shape_init=shape_init, biattention=True,
                                                          decoder_activation=decoder_activation)
        else:
            self.decoder = AdditiveModelDecoder(n_features=n_features, n_bins=n_bins, emsize=emsize, hidden_size=decoder_hidden_size, n_out=n_out,
                                                embed_dim=decoder_embed_dim, decoder_type=decoder_type,
                                                decoder_hidden_layers=decoder_hidden_layers, nhead=nhead, biattention=True,
                                                decoder_activation=decoder_activation, shape_init=shape_init)

        if decoder_type in ["special_token", "special_token_simple"]:
            self.token_embedding = nn.Parameter(torch.randn(1, 1, emsize))
        if self.input_layer_norm:
            self.input_norm = nn.LayerNorm(normalized_shape=(n_bins))
        self.init_weights()

    def init_weights(self):
        if self.init_method is not None:
            self.apply(get_init_method(self.init_method))
        if self.tabpfn_zero_weights:
            for bilayer in self.layers:
                for layer in [bilayer.cross_feature_attention, bilayer.cross_sample_attention]:
                    nn.init.zeros_(layer.linear2.weight)
                    nn.init.zeros_(layer.linear2.bias)
                    attns = layer.self_attn if isinstance(layer.self_attn, nn.ModuleList) else [layer.self_attn]
                    for attn in attns:
                        nn.init.zeros_(attn.out_proj.weight)
                        nn.init.zeros_(attn.out_proj.bias)

    def _determine_is_categorical(self, x_src_org: torch.Tensor) -> torch.Tensor:
        MAX_NUM_CATEGORIES = 100  # --> Categorical prior
        sequence_length, batch_size, num_features = x_src_org.shape

        # Preallocate the is_categorical tensor with the correct shape
        is_categorical = torch.zeros((1, batch_size, num_features), device=x_src_org.device)

        # Loop over features, but vectorize over batches
        for feature_i in range(num_features):
            # Extract all values for the current feature across all sequences and batches
            feature_values = x_src_org[:, :, feature_i]

            # Use broadcasting to compare each element with each other element
            # This creates a mask of shape (sequence_length, sequence_length, batch_size)
            # where each slice along the last dimension indicates unique values
            unique_mask = feature_values.unsqueeze(0) != feature_values.unsqueeze(1)

            # Sum over the sequence_length dimension to count unique values
            # The result has shape (sequence_length, batch_size)
            unique_counts = unique_mask.sum(dim=0)

            # Since we are counting non-equalities, the diagonal will be all False
            # We need to add 1 to each count to account for the diagonal
            unique_counts += 1

            # Determine if the feature is categorical for each batch element
            # The result has shape (batch_size,)
            is_categorical_feature = unique_counts.max(dim=0).values < MAX_NUM_CATEGORIES

            # Assign the result to the appropriate slice of the is_categorical tensor
            is_categorical[0, :, feature_i] = is_categorical_feature

        # Now is_categorical indicates for each feature and batch element whether it is categorical
        return is_categorical.to(torch.float32)

    def forward(self, src, single_eval_pos=None):
        assert isinstance(src, tuple), 'inputs (src) have to be given as (x,y) or (style,x,y) tuple'

        _, x_src_org, y_src_org = src

        X_onehot, _ = bin_data(x_src_org, n_bins=self.n_bins, nan_bin=self.nan_bin,
                               single_eval_pos=single_eval_pos)
        X_onehot = X_onehot.float()
        if self.fourier_features > 0:
            x_fourier = get_fourier_features(x_src_org, self.fourier_features)
            X_features = torch.cat([X_onehot, x_fourier], -1)
        else:
            X_features = X_onehot

        if self.input_layer_norm:
            X_features = self.input_norm(X_features)

        x_src = self.encoder(X_features)
        
        if hasattr(self, 'categorical_embedding'):
            # Determine which feature in each batch is categorical
            is_categorical = self._determine_is_categorical(x_src_org)  # (1, batch_size, num_features)
            x_src += self.categorical_embedding(is_categorical)
        if self.y_encoder is None:
            enc_train = x_src[:single_eval_pos]
        else:
            y_src = self.y_encoder(y_src_org.unsqueeze(-1) if len(y_src_org.shape) < len(x_src.shape) else y_src_org)
            enc_train = x_src[:single_eval_pos] + y_src[:single_eval_pos].unsqueeze(-2)

        # FIXME Refactor into a function to share with mothernet
        if self.decoder_type in ["special_token", "special_token_simple"]:
            enc_train = torch.cat([self.token_embedding.repeat(1, enc_train.shape[1], 1), enc_train], 0)
        elif self.decoder_type == "class_tokens":
            if not isinstance(self.y_encoder, OneHotAndLinear):
                raise ValueError("class_tokens decoder type is only supported with OneHotAndLinear y_encoder")
            repeated_class_tokens = self.y_encoder.weight.T.unsqueeze(1).repeat(1, enc_train.shape[1], 1)
            enc_train = torch.cat([repeated_class_tokens, enc_train], 0)

        if self.input_ln is not None:
            enc_train = self.input_ln(enc_train)
        output = enc_train
        if len(self.layers):
            for mod in self.layers:
                output = mod(output, src_mask=single_eval_pos)

            weights, biases = self.decoder(output, y_src_org[:single_eval_pos])

        if self.marginal_residual:
            class_averages = self.class_average_layer(X_onehot[:single_eval_pos], y_src_org[:single_eval_pos])
            # class averages are batch x outputs x features x bins
            # output is batch x features x bins x outputs
            marginals = self.marginal_residual_layer(class_averages).permute(0, 2, 3, 1)
            if len(self.layers):
                weights = weights + marginals
            else:
                weights = marginals
                biases = None

        # n samples, b batch, k feature, d bins, o outputs
        h = torch.einsum("nbkd,bkdo->nbo", X_onehot[single_eval_pos:], weights)
        if self.factorized_output:
            h += biases
        else:
            assert biases is None

        if h.isnan().all():
            print("NAN")
            import pdb
            pdb.set_trace()
        return h

