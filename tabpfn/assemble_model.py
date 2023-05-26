from torch import nn
from tabpfn.transformer import TransformerModel
from tabpfn.transformer_make_model import TransformerModelMaker, TransformerModelMakeMLP


def assemble_model(encoder_generator, num_features, emsize, nhead, nhid, nlayers, dropout, y_encoder, input_normalization,
                   model_maker, criterion, efficient_eval_masking=False,
                   output_attention=False, special_token=False, predicted_hidden_layer_size=None, decoder_embed_dim=None,
                   decoder_hidden_size=None, decoder_two_hidden_layers=False, no_double_embedding=False,
                   load_weights_from_this_state_dict=None, load_model_strict=True, verbose=False, **model_extra_args):
    encoder = encoder_generator(num_features, emsize)
    #style_def = dl.get_test_batch()[0][0] # the style in batch of the form ((style, x, y), target, single_eval_pos)
    style_def = None
    #print(f'Style definition of first 3 examples: {style_def[:3] if style_def is not None else None}')
    if isinstance(criterion, nn.GaussianNLLLoss):
        n_out = 2
    elif isinstance(criterion, nn.CrossEntropyLoss):
        n_out = criterion.weight.shape[0]
    else:
        n_out = 1
    if model_maker == "mlp":
        model = TransformerModelMakeMLP(encoder, n_out, emsize, nhead, nhid, nlayers, dropout,
                                y_encoder=y_encoder, input_normalization=input_normalization,
                                efficient_eval_masking=efficient_eval_masking, output_attention=output_attention, special_token=special_token,
                                predicted_hidden_layer_size=predicted_hidden_layer_size, decoder_embed_dim=decoder_embed_dim,
                                decoder_hidden_size=decoder_hidden_size, decoder_two_hidden_layers=decoder_two_hidden_layers,
                                no_double_embedding=no_double_embedding,
                                **model_extra_args
                                )
    elif model_maker:
        model = TransformerModelMaker(encoder, n_out, emsize, nhead, nhid, nlayers, dropout,
                                y_encoder=y_encoder, input_normalization=input_normalization,
                                efficient_eval_masking=efficient_eval_masking, **model_extra_args
                                )
    else:
        model = TransformerModel(encoder, n_out, emsize, nhead, nhid, nlayers, dropout,
                                y_encoder=y_encoder, input_normalization=input_normalization,
                                efficient_eval_masking=efficient_eval_masking, **model_extra_args
                                )
    model.criterion = criterion
    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict, strict=load_model_strict)

    if verbose:
        print(f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters")


    return model
