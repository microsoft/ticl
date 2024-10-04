import argparse
from ticl.config_utils import str2bool
from ticl.model_configs import get_model_default_config


class GroupedArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs['formatter_class'] = argparse.ArgumentDefaultsHelpFormatter
        super().__init__(*args, **kwargs)
    # This extends the argparse.ArgumentParser to allow for nested namespaces via groups
    # nesting of groups is done by giving them names with dots in them

    def parse_known_args(self, args=None, namespace=None):
        results, args = super().parse_known_args(args=args, namespace=namespace)
        nested_by_groups = argparse.Namespace()
        for group in self._action_groups:
            # group could have been created if we saw a nested group first
            new_subnamespace = getattr(nested_by_groups, group.title, argparse.Namespace())
            for action in group._group_actions:
                if action.dest is not argparse.SUPPRESS and hasattr(results, action.dest):
                    setattr(new_subnamespace, action.dest, getattr(results, action.dest))
            if new_subnamespace != argparse.Namespace():
                parts = group.title.split(".")
                parent_namespace = nested_by_groups
                for part in parts[:-1]:
                    if not hasattr(parent_namespace, part):
                        setattr(parent_namespace, part, argparse.Namespace())
                    parent_namespace = getattr(parent_namespace, part)
                setattr(parent_namespace, parts[-1], new_subnamespace)

        return nested_by_groups, args


def make_model_level_argparser(description="Train transformer-style model on synthetic data"):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(required=True, parser_class=GroupedArgParser,
                                       description="Choose the model type to train.", dest='model_type')
    mothernet_parser = subparsers.add_parser('mothernet', help='Train a mothernet model')
    mothernet_parser.set_defaults(model_type='mothernet')
    mothernet_parser = argparser_from_config(description="Train Mothernet", parser=mothernet_parser)

    tabpfn_parser = subparsers.add_parser('tabpfn', help='Train a tabpfn model')
    tabpfn_parser.set_defaults(model_type='tabpfn')
    tabpfn_parser = argparser_from_config(description="Train tabpfn", parser=tabpfn_parser)

    additive_parser = subparsers.add_parser('additive', help='Train an additive mothernet model')
    additive_parser.set_defaults(model_type='additive')
    additive_parser = argparser_from_config(description="Train additive", parser=additive_parser)

    perceiver_parser = subparsers.add_parser('perceiver', help='Train a perceiver mothernet model')
    perceiver_parser.set_defaults(model_type='perceiver')
    perceiver_parser = argparser_from_config(description="Train perceiver", parser=perceiver_parser)

    batabpfn_parser = subparsers.add_parser('batabpfn', help='Train a bi-attention tabpfn model')
    batabpfn_parser.set_defaults(model_type='batabpfn')
    batabpfn_parser = argparser_from_config(description="Train batabpfn", parser=batabpfn_parser)

    baam_parser = subparsers.add_parser('baam', help='Train a bi-attention additive mothernet model')
    baam_parser.set_defaults(model_type='baam')
    baam_parser = argparser_from_config(description="Train baam", parser=baam_parser)
    
    ssm_tabpfn_parser = subparsers.add_parser('ssm_tabpfn', help='Train a SSMtabpfn model')
    ssm_tabpfn_parser.set_defaults(model_type='ssm_tabpfn')
    ssm_tabpfn_parser = argparser_from_config(description="Train SSMtabpfn", parser=ssm_tabpfn_parser)

    ssm_mothernet_parser = subparsers.add_parser('ssm_mothernet', help='Train a ssm_mothernet model')
    ssm_mothernet_parser.set_defaults(model_type='ssm_mothernet')
    ssm_mothernet_parser = argparser_from_config(description="Train SSMMothernet", parser=ssm_mothernet_parser)

    return parser


def argparser_from_config(parser, description="Train Mothernet"):
    model_type = parser.get_default('model_type')
    config = get_model_default_config(model_type)
    # all models have general, optimizer, dataloader and transformer parameters
    general = parser.add_argument_group('general')
    general.add_argument('-g', '--gpu-id', type=int, help='GPU id')
    general.add_argument('-C', '--use-cpu', help='whether to use cpu', action='store_true')

    optimizer = parser.add_argument_group('optimizer')
    optimizer.add_argument('-E', '--epochs', type=int, help='number of epochs')
    optimizer.add_argument('-l', '--learning-rate', type=float, help='maximum learning rate')
    optimizer.add_argument('-k', '--aggregate_k_gradients', type=int, help='number steps to aggregate gradient over')
    optimizer.add_argument('-A', '--adaptive-batch-size', help='Wether to progressively increase effective batch size.',
                           type=str2bool)
    optimizer.add_argument('-w', '--weight-decay', type=float, help='Weight decay for AdamW.')
    optimizer.add_argument('-Q', '--learning-rate-schedule', help="Learning rate schedule. Cosine, constant or exponential")
    optimizer.add_argument('-U', '--warmup-epochs', type=int, help="Number of epochs to warm up learning rate (linear climb)")
    optimizer.add_argument('-t', '--train-mixed-precision', help='whether to train with mixed precision', type=str2bool)
    optimizer.add_argument('--adam-beta1', type=float)
    optimizer.add_argument('--lr-decay', help="learning rate decay when using exponential schedule", type=float)
    optimizer.add_argument('--min-lr', help="minimum learning rate for any schedule", type=float)
    optimizer.add_argument('--reduce-lr-on-spike', help="Whether to half learning rate when observing a loss spike", type=str2bool)
    optimizer.add_argument('--spike-tolerance', help="how many times the std makes it a spike", type=int)
    optimizer.set_defaults(**config['optimizer'])

    dataloader = parser.add_argument_group('dataloader')
    dataloader.add_argument('-b', '--batch-size', type=int, help='physical batch size')
    dataloader.add_argument('-n', '--num-steps', type=int, help='number of steps per epoch')
    dataloader.add_argument('--min-eval-pos', type=int, help='minimum evaluation position')
    dataloader.add_argument('--random-n-samples', type=int, help='whether to sample n_samples randomly')
    dataloader.add_argument('--n-test-samples', type=int, help='number of test samples')
    dataloader.set_defaults(**config['dataloader'])
    
    openmlloader = parser.add_argument_group('openmlloader')
    openmlloader.add_argument('--valid-data', default='new', help='whether to use large dataset', choices = ['new', 'large', 'old'])
    openmlloader.add_argument('--pca', default = False, help='whether to use pca', action = 'store_true')
    openmlloader.set_defaults(**config['openmlloader'])


    if 'transformer' in config:
        transformer = parser.add_argument_group('transformer')
        transformer.add_argument('-e', '--emsize', type=int, help='embedding size')
        transformer.add_argument('-N', '--nlayers', type=int, help='number of transformer layers')
        transformer.add_argument('--init-method', help='Weight initialization method.')
        transformer.add_argument('--y-encoder', help='Encoder for labels. "linear", "onehot" or None.')
        transformer.add_argument('--tabpfn-zero-weights', help='Whether to use zeroing of weights from tabpfn code.', type=str2bool)
        transformer.add_argument('--pre-norm', action='store_true')
        transformer.add_argument('--classification-task', type=str2bool, help='Whether to use classification or regression.')
        # transformer.add_argument('--model', type = str, choices = ['standard_attention', 'flash_attention'], help = 'which ssm model to use')
        # transformer.add_argument('--causal-mask', help='Whether to use causal attention', action='store_true', default=False)
        transformer.set_defaults(**config['transformer'])
    elif 'ssm' in config:
        ssm = parser.add_argument_group('ssm')
        ssm.add_argument('-e', '--emsize', type=int, help='embedding size')
        ssm.add_argument('-N', '--nlayers', type=int, help='number of transformer layers')
        ssm.add_argument('--init-method', help='Weight initialization method.')
        ssm.add_argument('--y-encoder', help='Encoder for labels. "linear", "onehot" or None.')
        ssm.add_argument('--tabpfn-zero-weights', help='Whether to use zeroing of weights from tabpfn code.', type=str2bool)
        ssm.add_argument('--pre-norm', action='store_true')
        ssm.add_argument('--classification-task', type=str2bool, help='Whether to use classification or regression.')
        ssm.add_argument('--model', type = str, choices = ['mamba1', 'mamba2', 'linear_attention', 'fla'], help = 'which ssm model to use')
        
        ## specific to fla
        ssm.add_argument('--feature-map', help='when the model is fla, which feature map to use', type = str, choices = ['identity', 'elu'])
        ssm.add_argument('--norm-output', help='when the model is fla, whether to normalize the output of the model', action = 'store_true', default = False)
        ssm.add_argument('--causal-mask', help='when the model is fla, Whether to use causal attention', action='store_true', default=False)
        ssm.set_defaults(**config['ssm'])
    else:
        raise ValueError("No transformer or ssm config found in model config.")

    if model_type in ['baam', 'batabpfn']:
        biattention = parser.add_argument_group('biattention')
        biattention.add_argument('--input-embedding', type=str, help='input embedding type')
        biattention.set_defaults(**config['biattention'])

    if model_type in ['mothernet', 'additive', 'baam', 'perceiver', 'ssm_mothernet']:
        mothernet = parser.add_argument_group('mothernet')
        mothernet.add_argument('-d', '--decoder-embed-dim', type=int, help='decoder embedding size')
        mothernet.add_argument('-H', '--decoder-hidden-size', type=int, help='decoder hidden size')
        mothernet.add_argument('--decoder-activation', type=str, help='decoder activation')
        mothernet.add_argument('-D', '--decoder-type',
                               help="Decoder Type. 'output_attention', 'special_token', 'class_average' or 'average'.", type=str)
        mothernet.add_argument('-T', '--decoder-hidden-layers', help='How many hidden layers to use in decoder MLP', type=int)
        mothernet.add_argument('-P', '--predicted-hidden-layer-size', type=int, help='Size of hidden layers in predicted network.')
        mothernet.add_argument('-L', '--predicted-hidden-layers', type=int, help='number of predicted hidden layers')
        mothernet.add_argument('--predicted-activation', type=str, help="activation in predicted network")
        mothernet.add_argument('-r', '--low-rank-weights', type=str2bool, help='Whether to use low-rank weights in mothernet.')
        mothernet.add_argument('-W', '--weight-embedding-rank', type=int, help='Rank of weights in predicted network.')
        mothernet.set_defaults(**config['mothernet'])

    if model_type in ['additive', 'baam']:
        additive = parser.add_argument_group('additive')
        additive.add_argument('--input-bin-embedding',
                              help="'linear' for linear bin embedding, 'non-linear' for nonlinear, 'none' or False for no embedding.", type=str)
        additive.add_argument('--bin-embedding-rank', help="Rank of bin embedding", type=int)
        additive.add_argument('--fourier-features', help="Number of Fourier features to add per feature. A value of 0 means off.", type=int)
        additive.add_argument('--n-bins', help="Number of bins", type=int)
        additive.add_argument('--nan-bin', help="Whether to use the last bin to denote a nan value.", type=str2bool)
        additive.add_argument('--sklearn-binning', help="Whether to bin the features with less num bins features using sklearn method.", type=str2bool)
        additive.add_argument('--categorical-embedding', help="Whether to embed the categorical features using a separate embedding", type=str2bool)
        additive.add_argument('--marginal-residual', help="Whether to learn the residual of the marginals. 'output', 'decoder' or 'none'.", type=str)
        additive.add_argument('--factorized-output', help="whether to use a factorized output", type=str2bool)
        additive.add_argument('--output-rank', help="Rank of output in factorized output", type=int)
        additive.add_argument('--input-layer-norm', help="Whether to use layer norm on one-hot encoded data.", type=str2bool)
        additive.add_argument('--shape-attention', help="Whether to use attention in low rank output.", type=str2bool)
        additive.add_argument('--shape-attention-heads', help="Number of heads in shape attention.", type=int)
        additive.add_argument('--n-shape-functions', help="Number of shape functions in shape attention.", type=int)
        additive.add_argument('--shape-init', help="How to initialize shape functions. 'constant' for unit variance, 'inverse' for 1/(n_shape_functions * n_bins), "
                              "'sqrt' for 1/sqrt(n_shape_functions * n_bins). 'inverse_bins' for 1/n_bins, 'inverse_sqrt_bins' for 1/sqrt(n_bins)",
                              type=str)
        additive.set_defaults(**config['additive'])

    if model_type in ['perceiver']:
        perceiver = parser.add_argument_group('perceiver')
        perceiver.add_argument('--num-latents', help="number of latent variables in perceiver", type=int)
        # perceiver.add_argument('--perceiver-large-dataset', action='store_true')
        perceiver.set_defaults(**config['perceiver'])


    # Prior and data generation
    prior = parser.add_argument_group('prior')
    prior.add_argument('--num-features', help="Maximum number of features in prior", type=int)
    prior.add_argument('--n-samples', help="Maximum Number of samples in prior", type=int)
    prior.add_argument('--prior-type', help="Which prior to use, available ['prior_bag', 'boolean_only', 'bag_boolean', 'step_function'].", type=str)
    prior.set_defaults(**config['prior'])

    classification_prior = parser.add_argument_group('prior.classification')
    classification_prior.add_argument('--multiclass-type', help="Which multiclass prior to use ['steps', 'rank'].", type=str)
    classification_prior.add_argument('--num-features-sampler', help="How to sample number of features, 'uniform', 'double_sample'. ", type=str)
    classification_prior.add_argument('--multiclass-max-steps', help="Maximum number of steps in multiclass step prior", type=int)
    classification_prior.add_argument('--pad-zeros', help="Whether to pad data with zeros for consistent size", type=str2bool)
    classification_prior.add_argument('--max-num-classes', help="Maximum number of classes. 0 means regression.", type=int)
    classification_prior.add_argument('--nan-prob-no-reason', help="NaN probability missing at random.", type=float)
    classification_prior.add_argument('--nan-prob-a-reason', help="NaN probability missing not at random.", type=float)
    classification_prior.add_argument('--categorical-feature-p', help="Categorical feature probability.", type=float)
    classification_prior.add_argument('--feature-curriculum', help="Whether to use a curriculum for number of features", type=str2bool)
    classification_prior.set_defaults(**config['prior']['classification'])

    mlp_prior = parser.add_argument_group('prior.mlp')
    mlp_prior.add_argument('--add-uninformative-features', help="Whether to add uniformative features in the MLP prior.", type=str2bool)
    mlp_prior.set_defaults(**config['prior']['mlp'])

    boolean = parser.add_argument_group('prior.boolean')
    boolean.add_argument('--p-uninformative', help="Probability of adding uninformative features in boolean prior",
                         type=float)
    boolean.add_argument('--max-fraction-uninformative', help="Maximum fraction opf uninformative features in boolean prior",
                         type=float)
    boolean.set_defaults(**config['prior']['boolean'])

    # serialization, loading, logging
    orchestration = parser.add_argument_group('orchestration')
    orchestration.add_argument('--extra-fast-test', help="whether to use tiny data", action='store_true')
    orchestration.add_argument('--stop-after-epochs', help="for pausing rungs with synetune", type=int, default=None)
    orchestration.add_argument('--seed-everything', help="whether to seed everything for testing and benchmarking", default = True, type=str2bool)
    orchestration.add_argument('--experiment', help="Name of mlflow experiment", default='Default')
    orchestration.add_argument('-R', '--create-new-run', help="Create as new MLFLow run, even if continuing", action='store_true')
    orchestration.add_argument('-B', '--base-path', default='.')
    orchestration.add_argument('--save-every', default=10, type=int)
    orchestration.add_argument('--st_checkpoint_dir', help="checkpoint dir for synetune", type=str, default=None)
    orchestration.add_argument('--use-mlflow', help="whether to use mlflow", action='store_true')
    orchestration.add_argument('--use-wandb', help="whether to use wandb", action='store_true')
    orchestration.add_argument('-f', '--warm-start-from', help='Warm start from this file')
    orchestration.add_argument('-c', '--continue-run', help='Whether to read the old config when warm starting', action='store_true')
    orchestration.add_argument('-s', '--load-strict', help='Whether to load the architecture strictly when warm starting', action='store_true')
    orchestration.add_argument('--restart-scheduler', help='Whether to restart the scheduler when warm starting', action='store_true')
    orchestration.add_argument('--detect-anomaly', help='Whether enable anomaly detection in pytorch. For debugging only.', action='store_true')
    orchestration.add_argument('--validate', type=str2bool, help='Whether to perform validation.', default=True)
    orchestration.add_argument('--progress-bar', type=str2bool, help='Whether to show a progress bar.', default=False)
    orchestration.add_argument('--wandb-overwrite', help='Whether to overwrite wandb runs.', action='store_true', default=False)

    # orchestration options are not part of the default config
    return parser
