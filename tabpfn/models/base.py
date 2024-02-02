import torch.nn as nn
from tabpfn.models import encoders

class BaseTabularModel(nn.Module):

    def get_encoder(self):
        if (('nan_prob_no_reason' in config and config['nan_prob_no_reason'] > 0.0) or
            ('nan_prob_a_reason' in config and config['nan_prob_a_reason'] > 0.0) or
                ('nan_prob_unknown_reason' in config and config['nan_prob_unknown_reason'] > 0.0)):
            encoder = encoders.NanHandlingEncoder(num_features, config_transformer['emsize'])
        else:
            encoder = encoders.Linear(num_features, config_transformer['emsize'], replace_nan_by_zero=True)

        if 'encoder' in config and config['encoder'] == 'featurewise_mlp':
            encoder = encoders.FeaturewiseMLP
        return encoder


    def get_y_encoder(self, y_encoder):
        if y_encoder == 'one_hot':
            y_encoder = encoders.OneHotAndLinear(config['prior']['classification']['max_num_classes'], emsize=config['transformer']['emsize'])
        elif y_encoder == 'linear':
            y_encoder = encoders.Linear(1, emsize=self.emsize)
        else:
            raise ValueError(f"Unknown y_encoder: {y_encoder}")
        return y_encoder
