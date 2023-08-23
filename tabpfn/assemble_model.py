


def assemble_model(encoder_generator, num_features, emsize, nhead, nhid, nlayers, dropout, y_encoder, input_normalization,
                   model_maker, max_num_classes, efficient_eval_masking=False,
                   output_attention=False, special_token=False, predicted_hidden_layer_size=None, decoder_embed_dim=None,
                   decoder_hidden_size=None, decoder_two_hidden_layers=False, no_double_embedding=False,
                   model_state=None, load_model_strict=True, verbose=False, pre_norm=False, predicted_hidden_layers=1, weight_embedding_rank=None, **model_extra_args):
    encoder = encoder_generator(num_features, emsize)
    decoder_hidden_size = decoder_hidden_size or nhid

    from tabpfn.transformer import TransformerModel
    from tabpfn.perceiver import TabPerceiver
    from tabpfn.transformer_make_model import TransformerModelMaker, TransformerModelMakeMLP

    if max_num_classes > 2:
        n_out = max_num_classes
    else:
        n_out = 1
    if model_maker == "mlp":
        model = TransformerModelMakeMLP(encoder, n_out, emsize, nhead, nhid, nlayers, dropout,
                                y_encoder=y_encoder, input_normalization=input_normalization,
                                efficient_eval_masking=efficient_eval_masking, output_attention=output_attention, special_token=special_token,
                                predicted_hidden_layer_size=predicted_hidden_layer_size, decoder_embed_dim=decoder_embed_dim,
                                decoder_hidden_size=decoder_hidden_size, decoder_two_hidden_layers=decoder_two_hidden_layers,
                                no_double_embedding=no_double_embedding, pre_norm=pre_norm, predicted_hidden_layers=predicted_hidden_layers, weight_embedding_rank=weight_embedding_rank,
                                **model_extra_args
                                )
    elif model_maker == 'perceiver':
        model = TabPerceiver(encoder=encoder, input_dim=emsize, depth=nlayers, n_out=n_out, latent_dim=emsize, latent_heads=nhead, ff_dropout=dropout,
                                y_encoder=y_encoder, output_attention=output_attention, special_token=special_token,
                                predicted_hidden_layer_size=predicted_hidden_layer_size, decoder_embed_dim=decoder_embed_dim,
                                decoder_hidden_size=decoder_hidden_size, decoder_two_hidden_layers=decoder_two_hidden_layers,
                                no_double_embedding=no_double_embedding, predicted_hidden_layers=predicted_hidden_layers, weight_embedding_rank=None,
                                **model_extra_args
                                )
    else:
        model = TransformerModel(encoder, n_out, emsize, nhead, nhid, nlayers, dropout,
                                y_encoder=y_encoder, input_normalization=input_normalization,
                                efficient_eval_masking=efficient_eval_masking, pre_norm=pre_norm, **model_extra_args
                                )
    if model_state is not None:
        model.load_state_dict(model_state, strict=load_model_strict)

    if verbose:
        print(f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters")


    return model
