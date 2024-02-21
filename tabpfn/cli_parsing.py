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


def argparser_from_config(config, description="Train Mothernet"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-g', '--gpu-id', type=int, help='GPU id')
    parser.add_argument('-e', '--em-size', type=int, help='embedding size', default=512, dest='emsize')
    parser.add_argument('-n', '--num-steps', type=int, help='number of steps per epoch')
    parser.add_argument('-E', '--epochs', type=int, help='number of epochs', default=4000)
    parser.add_argument('-l', '--learning-rate', type=float, help='maximum learning rate', default=0.00003, dest='lr')
    parser.add_argument('-N', '--nlayers', type=int, help='number of transformer layers', default=12)
    parser.add_argument('-k', '--agg-gradients', type=int, help='number steps to aggregate gradient over', default=1, dest='aggregate_k_gradients')
    parser.add_argument('-b', '--batch-size', type=int, help='physical batch size', default=8)
    parser.add_argument('-A', '--adaptive-batch-size', help='Wether to progressively increase effective batch size.', default=True, type=str2bool)
    parser.add_argument('-w', '--weight-decay', type=float, help='Weight decay for AdamW.', default=0)
    parser.add_argument('-Q', '--learning-rate-schedule', help="Learning rate schedule. Cosine, constant or exponential", default='cosine')
    parser.add_argument('-U', '--warmup-epochs', type=int, help="Number of epochs to warm up learning rate (linear climb)", default=20)
    parser.add_argument('-t', '--train-mixed-precision', help='whether to train with mixed precision', default=True, type=str2bool)
    parser.add_argument('--adam-beta1', default=0.9, type=float)
    parser.add_argument('-C', '--use-cpu', help='whether to use cpu', action='store_true')
    parser.add_argument('--lr-decay', help="learning rate decay when using exponential schedule", default=0.99, type=float)
    parser.add_argument('--min-lr', help="minimum learning rate for any schedule", default=1e-8, type=float)
    parser.add_argument('--pre-norm', action='store_true')
    parser.add_argument('--reduce-lr-on-spike', help="Whether to half learning rate when observing a loss spike", default=False, type=str2bool)
    parser.add_argument('--spike-tolerance', help="how many times the std makes it a spike", default=4, type=int)

    # selecting model
    parser.add_argument('-m', '--model-type', type=str, help='model maker kind. mlp for mothernet, perceiver, additive, or False for TabPFN', default='mlp')

    # Mothernet specific
    parser.add_argument('-d', '--decoder-em-size', type=int, help='decoder embedding size', default=1024, dest='decoder_embed_dim')
    parser.add_argument('-H', '--decoder-hidden-size', type=int, help='decoder hidden size', default=2048)
    
    parser.add_argument('-D', '--no-double-embedding', help='whether to reuse transformer embedding for mlp', action='store_false')
    parser.add_argument('-S', '--special-token',
                        help='whether add a special output token in the first layer as opposed to having one in the last attention layer. If True, decoder-em-size is ignored.', default=False, type=str2bool)
    parser.add_argument('-T', '--decoder-two-hidden-layers', help='whether to use two hidden layers for the decoder', default=False, type=str2bool)
    parser.add_argument('-P', '--predicted-hidden-layer-size', type=int, help='Size of hidden layers in predicted network.', default=128)
    parser.add_argument('-L', '--num-predicted-hidden-layers', type=int, help='number of predicted hidden layers', default=1, dest='predicted_hidden_layers')
    parser.add_argument('-r', '--low-rank-weights', type=str2bool, help='Whether to use low-rank weights in mothernet.', default=True)
    parser.add_argument('-W', '--weight-embedding-rank', type=int, help='Rank of weights in predicted network.', default=32)

    # Additive model (WIP)
    parser.add_argument('--input-bin-embedding', help="'linear' for linear bin embedding, 'non-linear' for nonlinear, 'none' or False for no embedding.", type=str, default="none")
    parser.add_argument('--bin-embedding-rank', help="Rank of bin embedding", type=int, default=16)
    parser.add_argument('--factorized-output', help="whether to use a factorized output", type=str2bool, default=False)
    parser.add_argument('--output-rank', help="Rank of output in factorized output", type=int, default=16)

    # Perceiver (even more WIP)
    parser.add_argument('--num-latents', help="number of latent variables in perceiver", default=512, type=int)
    parser.add_argument('--perceiver-large-dataset', action='store_true')

    # Prior and data generation
    parser.add_argument('--extra-fast-test', help="whether to use tiny data", action='store_true')
    parser.add_argument('--multiclass-type', help="Which multiclass prior to use ['steps', 'rank'].", default='rank', type=str)
    parser.add_argument('--prior-type', help="Which prior to use, available ['prior_bag', 'boolean_only', 'bag_boolean'].", default='prior_bag', type=str)
    parser.add_argument('--add-uninformative-features', help="Whether to add uniformative features in the MLP prior.", default=False, type=str2bool)
    parser.add_argument('--heterogeneous-batches', help="Whether to resample MLP hypers for each sample instead of each batch.", default='False', type=str2bool)

    parser.add_argument('--boolean-p-uninformative', help="Probability of adding uninformative features in boolean prior", default=0.5, type=float)
    parser.add_argument('--boolean-max-fraction-uninformative', help="Maximum fraction opf uninformative features in boolean prior", default=0.5, type=float)
    
    # serialization, loading, logging
    parser.add_argument('--stop-after-epochs', help="for pausing rungs with synetune", type=int, default=None)
    parser.add_argument('--seed-everything', help="whether to seed everything for testing and benchmarking", action='store_true')
    parser.add_argument('--experiment', help="Name of mlflow experiment", default='Default')
    parser.add_argument('-R', '--create-new-run', help="Create as new MLFLow run, even if continuing", action='store_true')
    parser.add_argument('-B', '--base-path', default='.')
    parser.add_argument('--save-every', default=10, type=int)
    parser.add_argument('--st_checkpoint_dir', help="checkpoint dir for synetune", type=str, default=None)
    parser.add_argument('--no-mlflow', help="whether to use mlflow", action='store_true')
    parser.add_argument('-f', '--load-file', help='Warm start from this file', dest='warm_start_from')
    parser.add_argument('-c', '--continue-run', help='Whether to read the old config when warm starting', action='store_true')
    parser.add_argument('-s', '--load-strict', help='Whether to load the architecture strictly when warm starting', action='store_true')
    parser.add_argument('--restart-scheduler', help='Whether to restart the scheduler when warm starting', action='store_true')
    return parser


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError("Boolean value expected.")
