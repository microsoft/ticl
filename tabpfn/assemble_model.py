

def assemble_model(encoder_layer, num_features, emsize, nhead, nhid, nlayers, dropout, y_encoder_layer, input_normalization,
                   model_type, max_num_classes, efficient_eval_masking=False,
                   output_attention=False, special_token=False, predicted_hidden_layer_size=None, decoder_embed_dim=None,
                   decoder_hidden_size=None, decoder_two_hidden_layers=False, no_double_embedding=False,
                   model_state=None, load_model_strict=True, verbose=False, pre_norm=False, predicted_hidden_layers=1, weight_embedding_rank=None, low_rank_weights=False, num_latents=512, input_bin_embedding=False,
                   factorized_output=False, output_rank=None, bin_embedding_rank=None, **model_extra_args):

    from tabpfn.models.mothernet_additive import MotherNetAdditive
    from tabpfn.models.perceiver import TabPerceiver
    from tabpfn.models.transformer import TabPFN
    from tabpfn.models.mothernet import MotherNet

    if max_num_classes > 2:
        n_out = max_num_classes
    else:
        n_out = 1
    if model_type == "mlp":
        model = MotherNet(
            encoder_layer, n_out=n_out, emsize=emsize, nhead=nhead, nhid=nhid, nlayers=nlayers, dropout=dropout,
            y_encoder_layer=y_encoder_layer, input_normalization=input_normalization,
            efficient_eval_masking=efficient_eval_masking, output_attention=output_attention, special_token=special_token,
            predicted_hidden_layer_size=predicted_hidden_layer_size, decoder_embed_dim=decoder_embed_dim,
            decoder_hidden_size=decoder_hidden_size, decoder_two_hidden_layers=decoder_two_hidden_layers,
            no_double_embedding=no_double_embedding, pre_norm=pre_norm, predicted_hidden_layers=predicted_hidden_layers, weight_embedding_rank=weight_embedding_rank, low_rank_weights=low_rank_weights,
            **model_extra_args
        )
    elif model_type == 'perceiver':
        model = TabPerceiver(
            encoder=encoder_layer, emsize=emsize, nlayers=nlayers, n_out=n_out, nhead=nhead, dropout=dropout,
            y_encoder_layer=y_encoder_layer, output_attention=output_attention, special_token=special_token,
            predicted_hidden_layer_size=predicted_hidden_layer_size, decoder_embed_dim=decoder_embed_dim,
            decoder_hidden_size=decoder_hidden_size, decoder_two_hidden_layers=decoder_two_hidden_layers,
            no_double_embedding=no_double_embedding, predicted_hidden_layers=predicted_hidden_layers, weight_embedding_rank=weight_embedding_rank,
            num_latents=num_latents, low_rank_weights=low_rank_weights,
            **model_extra_args
        )
    elif model_type == "additive":
        model = MotherNetAdditive(
            n_features=num_features, n_out=n_out, emsize=emsize, nhead=nhead, nhid=nhid, nlayers=nlayers, dropout=dropout, y_encoder=y_encoder,
            input_normalization=input_normalization, pre_norm=pre_norm, decoder_embed_dim=decoder_embed_dim,
            decoder_two_hidden_layers=decoder_two_hidden_layers, decoder_hidden_size=decoder_hidden_size, n_bins=64, input_bin_embedding=input_bin_embedding,
            factorized_output=factorized_output, output_rank=output_rank, bin_embedding_rank=bin_embedding_rank)
    elif model_type == "tabpfn":
        model = TabPFN(
            encoder_layer, n_out=n_out, emsize=emsize, nhead=nhead, nhid=nhid, nlayers=nlayers, dropout=dropout,
            y_encoder_layer=y_encoder_layer, input_normalization=input_normalization,
            efficient_eval_masking=efficient_eval_masking, pre_norm=pre_norm, **model_extra_args
        )
    else:
        raise ValueError(f"Unknown model type {model_type}.")
    return model
