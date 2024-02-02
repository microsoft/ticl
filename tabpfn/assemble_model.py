

def assemble_model(encoder_layer, y_encoder_layer, model_type, config_transformer, config_mothernet, config_additive, config_perceiver, num_features, max_num_classes):
    nhid = config_transformer['emsize'] * config_transformer['nhid_factor']
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
            encoder_layer, n_out=n_out, nhid=nhid,
            encoder_layer=encoder_layer, **config_transformer, **config_mothernet
        )
    elif model_type == 'perceiver':
        model = TabPerceiver(
            encoder_layer, n_out=n_out, nhid=nhid,
            y_encoder_layer=y_encoder_layer, **config_transformer, **config_mothernet, **config_perceiver
        )
    elif model_type == "additive":
        model = MotherNetAdditive(
            encoder_layer, n_out=n_out, nhid=nhid,
            y_encoder_layer=y_encoder_layer, **config_transformer, **config_mothernet, **config_additive)
    elif model_type == "tabpfn":
        model = TabPFN(
            encoder_layer, n_out=n_out, y_encoder_layer=y_encoder_layer, **config_transformer
        )
    else:
        raise ValueError(f"Unknown model type {model_type}.")
    return model
