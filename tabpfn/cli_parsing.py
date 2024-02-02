import argparse

class GroupedArgParser(argparse.ArgumentParser):
    # This extends the argparse.ArgumentParser to allow for nested namespaces via groups
    # nesting of groups is done by giving them names with dots in them

    def parse_args(self, argv):
        results = super().parse_args(argv)
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
                
        return nested_by_groups


def argparser_from_config(description="Train Mothernet"):
    parser = GroupedArgParser(description=description)

    general = parser.add_argument_group('general')
    general.add_argument('-m', '--model-type', type=str, help='model maker kind. mlp for mothernet, perceiver, additive, or False for TabPFN', default='mlp')
    general.add_argument('-g', '--gpu-id', type=int, help='GPU id')
    general.add_argument('-C', '--use-cpu', help='whether to use cpu', action='store_true')

    optimizer = parser.add_argument_group('optimizer')
    optimizer.add_argument('-E', '--epochs', type=int, help='number of epochs', default=4000)
    optimizer.add_argument('-l', '--learning-rate', type=float, help='maximum learning rate', default=0.00003)
    optimizer.add_argument('-k', '--agg-gradients', type=int, help='number steps to aggregate gradient over', default=1, dest='aggregate_k_gradients')
    optimizer.add_argument('-A', '--adaptive-batch-size', help='Wether to progressively increase effective batch size.', default=True, type=str2bool)
    optimizer.add_argument('-w', '--weight-decay', type=float, help='Weight decay for AdamW.', default=0)
    optimizer.add_argument('-Q', '--learning-rate-schedule', help="Learning rate schedule. Cosine, constant or exponential", default='cosine')
    optimizer.add_argument('-U', '--warmup-epochs', type=int, help="Number of epochs to warm up learning rate (linear climb)", default=20)
    optimizer.add_argument('-t', '--train-mixed-precision', help='whether to train with mixed precision', default=True, type=str2bool)
    optimizer.add_argument('--adam-beta1', default=0.9, type=float)
    optimizer.add_argument('--lr-decay', help="learning rate decay when using exponential schedule", default=0.99, type=float)
    optimizer.add_argument('--min-lr', help="minimum learning rate for any schedule", default=1e-8, type=float)
    optimizer.add_argument('--reduce-lr-on-spike', help="Whether to half learning rate when observing a loss spike", default=False, type=str2bool)
    optimizer.add_argument('--spike-tolerance', help="how many times the std makes it a spike", default=4, type=int)

    dataloader = parser.add_argument_group('dataloader')
    dataloader.add_argument('-b', '--batch-size', type=int, help='physical batch size', default=8)
    dataloader.add_argument('-n', '--num-steps', type=int, help='number of steps per epoch')


    transformer = parser.add_argument_group('transformer')
    transformer.add_argument('-e', '--em-size', type=int, help='embedding size', default=512, dest='emsize')
    transformer.add_argument('-N', '--nlayers', type=int, help='number of transformer layers', default=12)
    transformer.add_argument('--pre-norm', action='store_true')

    mothernet = parser.add_argument_group('mothernet')
    mothernet.add_argument('-d', '--decoder-em-size', type=int, help='decoder embedding size', default=1024, dest='decoder_embed_dim')
    mothernet.add_argument('-H', '--decoder-hidden-size', type=int, help='decoder hidden size', default=2048)
    
    mothernet.add_argument('-D', '--no-double-embedding', help='whether to reuse transformer embedding for mlp', action='store_false')
    mothernet.add_argument('-S', '--special-token',
                           help='whether add a special output token in the first layer as opposed to having one in the last attention layer. If True, decoder-em-size is ignored.', default=False, type=str2bool)
    mothernet.add_argument('-T', '--decoder-two-hidden-layers', help='whether to use two hidden layers for the decoder', default=False, type=str2bool)
    mothernet.add_argument('-P', '--predicted-hidden-layer-size', type=int, help='Size of hidden layers in predicted network.', default=128)
    mothernet.add_argument('-L', '--num-predicted-hidden-layers', type=int, help='number of predicted hidden layers', default=1, dest='predicted_hidden_layers')
    mothernet.add_argument('-r', '--low-rank-weights', type=str2bool, help='Whether to use low-rank weights in mothernet.', default=True)
    mothernet.add_argument('-W', '--weight-embedding-rank', type=int, help='Rank of weights in predicted network.', default=32)

    # Additive model (WIP)
    additive = parser.add_argument_group('additive')
    additive.add_argument('--shared-embedding', help="whether to use a shared low-rank embedding over bins in additive model", type=str2bool, default=False)

    # Perceiver
    perceiver = parser.add_argument_group('perceiver')
    perceiver.add_argument('--num-latents', help="number of latent variables in perceiver", default=512, type=int)
    # perceiver.add_argument('--perceiver-large-dataset', action='store_true')

    # Prior and data generation
    prior = parser.add_argument_group('prior')
    prior.add_argument('--prior-type', help="Which prior to use, available ['prior_bag', 'boolean_only', 'bag_boolean'].", default='prior_bag', type=str)
    classification_prior = parser.add_argument_group('prior.classification')
    classification_prior.add_argument('--multiclass-type', help="Which multiclass prior to use ['steps', 'rank'].", default='rank', type=str)

    mlp_prior = parser.add_argument_group('prior.mlp')
    mlp_prior.add_argument('--add-uninformative-features', help="Whether to add uniformative features in the MLP prior.", default=False, type=str2bool)
    prior.add_argument('--heterogeneous-batches', help="Whether to resample MLP hypers for each sample instead of each batch.", default='False', type=str2bool)
    boolean = parser.add_argument_group('prior.boolean')
    boolean.add_argument('--boolean-p-uninformative', help="Probability of adding uninformative features in boolean prior", default=0.5, type=float)
    boolean.add_argument('--boolean-max-fraction-uninformative', help="Maximum fraction opf uninformative features in boolean prior", default=0.5, type=float)
    
    # serialization, loading, logging
    orchestration = parser.add_argument_group('orchestration')
    orchestration.add_argument('--extra-fast-test', help="whether to use tiny data", action='store_true')
    orchestration.add_argument('--stop-after-epochs', help="for pausing rungs with synetune", type=int, default=None)
    orchestration.add_argument('--seed-everything', help="whether to seed everything for testing and benchmarking", action='store_true')
    orchestration.add_argument('--experiment', help="Name of mlflow experiment", default='Default')
    orchestration.add_argument('-R', '--create-new-run', help="Create as new MLFLow run, even if continuing", action='store_true')
    orchestration.add_argument('-B', '--base-path', default='.')
    orchestration.add_argument('--save-every', default=10, type=int)
    orchestration.add_argument('--st_checkpoint_dir', help="checkpoint dir for synetune", type=str, default=None)
    orchestration.add_argument('--no-mlflow', help="whether to use mlflow", action='store_true')
    orchestration.add_argument('-f', '--load-file', help='Warm start from this file', dest='warm_start_from')
    orchestration.add_argument('-c', '--continue-run', help='Whether to read the old config when warm starting', action='store_true')
    orchestration.add_argument('-s', '--load-strict', help='Whether to load the architecture strictly when warm starting', action='store_true')
    orchestration.add_argument('--restart-scheduler', help='Whether to restart the scheduler when warm starting', action='store_true')
    return parser


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError("Boolean value expected.")
