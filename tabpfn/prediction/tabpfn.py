import os
import pathlib
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, PowerTransformer, QuantileTransformer, RobustScaler
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from torch.utils.checkpoint import checkpoint

from tabpfn.model_builder import load_model
from tabpfn.utils import NOP, normalize_by_used_features_f, normalize_data, remove_outliers, to_ranking_low_mem


def _get_file(e, base_path, add_name, eval_addition):
    """
    Returns the different paths of model_file, model_path and results_file
    """
    model_file = f'models_diff/{add_name}_epoch_{e}.cpkt'
    model_path = os.path.join(base_path, model_file)
    results_file = os.path.join(base_path,
                                f'models_diff/prior_diff_real_results_{add_name}_n_0_epoch_{e}_{eval_addition}.pkl')
    return model_path, results_file


def _check_file(e, base_path, add_name, eval_addition, verbose):
    model_path, results_file = _get_file(e, base_path, add_name, eval_addition)
    if not Path(model_path).is_file():
        if add_name == "download":
            print('We have to download the TabPFN, as there is no checkpoint at ', model_path)
            print('It has about 100MB, so this might take a moment.')
            import requests
            url = 'https://github.com/automl/TabPFN/raw/main/tabpfn/models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt'
            r = requests.get(url, allow_redirects=True)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            open(model_path, 'wb').write(r.content)
        else:
            model_path = None
    else:
        if verbose:
            print(f"loading model from file {model_path}")
    return model_path, results_file


def load_model_workflow(e, add_name, base_path, device='cpu', eval_addition='', verbose=0):
    """
    Workflow for loading a model and setting appropriate parameters for diffable hparam tuning.
    """
    model_path = None
    if e == -1:
        for e_ in range(100, -1, -1):
            model_path_, results_file_ = _check_file(e_, base_path, add_name, eval_addition, verbose)
            if model_path_ is not None:
                e = e_
                model_path, results_file = model_path_, results_file_
                break
    else:
        model_path, results_file = _check_file(e, base_path, add_name, eval_addition, verbose)

    if model_path is None:
        model_path, results_file = _get_file(e, base_path, add_name, eval_addition)
        raise Exception('No checkpoint found at '+str(model_path))
    model, c = load_model(model_path, device, verbose=False)

    return model, c, results_file


class TabPFNClassifier(BaseEstimator, ClassifierMixin):

    models_in_memory = {}

    def __init__(self, device='cpu', base_path=pathlib.Path(__file__).parent.parent.resolve(), model_string='download',
                 N_ensemble_configurations=3, no_preprocess_mode=False, multiclass_decoder='permutation',
                 feature_shift_decoder=True, seed=0, no_grad=True, batch_size_inference=32,
                 subsample_features=False, verbose=False, scaling=True, epoch=-1):
        """
        Initializes the classifier and loads the model.
        Depending on the arguments, the model is either loaded from memory, from a file, or downloaded from the
        repository if no model is found.

        Can also be used to compute gradients with respect to the inputs X_train and X_test. Therefore no_grad has to be
        set to False and no_preprocessing_mode must be True. Furthermore, X_train and X_test need to be given as
        torch.Tensors and their requires_grad parameter must be set to True.


        :param device: If the model should run on cuda or cpu.
        :param base_path: Base path of the directory, from which the folders like models_diff can be accessed.
        :param model_string: Name of the model. Used first to check if the model is already in memory, and if not,
               tries to load a model with that name from the models_diff directory. It looks for files named as
               follows: "prior_diff_real_checkpoint" + model_string + "_n_0_epoch_e.cpkt", where e can be a number
               between 100 and 0, and is checked in a descending order.
        :param N_ensemble_configurations: The number of ensemble configurations used for the prediction. Thereby the
               accuracy, but also the running time, increases with this number.
        :param no_preprocess_mode: Specifies whether preprocessing is to be performed.
        :param multiclass_decoder: If set to permutation, randomly shifts the classes for each ensemble configuration.
        :param feature_shift_decoder: If set to true shifts the features for each ensemble configuration according to a
               random permutation.
        :param seed: Seed that is used for the prediction. Allows for a deterministic behavior of the predictions.
        :param batch_size_inference: This parameter is a trade-off between performance and memory consumption.
               The computation done with different values for batch_size_inference is the same,
               but it is split into smaller/larger batches.
        :param no_grad: If set to false, allows for the computation of gradients with respect to X_train and X_test.
               For this to correctly function no_preprocessing_mode must be set to true.
        :param subsample_features: If set to true and the number of features in the dataset exceeds self.max_features (100),
                the features are subsampled to self.max_features.
        """

        self.verbose = verbose
        self.device = device
        self.style = None
        self.N_ensemble_configurations = N_ensemble_configurations
        self.base__path = base_path
        self.base_path = base_path
        self.model_string = model_string
        self.no_preprocess_mode = no_preprocess_mode
        self.feature_shift_decoder = feature_shift_decoder
        self.multiclass_decoder = multiclass_decoder
        self.seed = seed
        self.no_grad = no_grad
        self.subsample_features = subsample_features
        self.epoch = epoch
        self.temperature = None
        self.scaling = scaling

        assert self.no_preprocess_mode if not self.no_grad else True, \
            "If no_grad is false, no_preprocess_mode must be true, because otherwise no gradient can be computed."

        self.batch_size_inference = batch_size_inference

    def remove_models_from_memory(self):
        self.models_in_memory = {}

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % len(cls)
            )

        self.classes_ = cls
        return np.asarray(y, dtype=np.float64, order="C")

    def fit(self, X, y, overwrite_warning=False):

        model_key = self.model_string+'|'+str(self.device) + '|' + str(self.epoch)
        if model_key in self.models_in_memory:
            if self.verbose:
                print(f"using model {model_key}")
            model, c, results_file = self.models_in_memory[model_key]
        else:
            model, c, results_file = load_model_workflow(self.epoch, add_name=self.model_string, base_path=self.base_path, device=self.device,
                                                         eval_addition='')
            self.models_in_memory[model_key] = (model, c, results_file)
            if len(self.models_in_memory) == 2:
                print('Multiple models in memory. This might lead to memory issues. Consider calling remove_models_from_memory()')
        self.c = c
        self.max_num_features = self.c['num_features']
        self.max_num_classes = self.c['max_num_classes']
        self.differentiable_hps_as_style = self.c['differentiable_hps_as_style']

        self.model = model
        if self.no_grad:
            # Check that X and y have correct shape
            X, y = check_X_y(X, y, force_all_finite=False)
        # Store the classes seen during fit
        y = self._validate_targets(y)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)

        self.X_ = X
        self.y_ = y

        if (X.shape[1] > self.max_num_features):
            if self.subsample_features:
                print('WARNING: The number of features for this classifier is restricted to ', self.max_num_features, ' and will be subsampled.')
            else:
                raise ValueError("The number of features for this classifier is restricted to ", self.max_num_features)
        if len(np.unique(y)) > self.max_num_classes:
            raise ValueError("The number of classes for this classifier is restricted to ", self.max_num_classes)
        if X.shape[0] > 1024 and not overwrite_warning:
            raise ValueError("⚠️ WARNING: TabPFN is not made for datasets with a trainingsize > 1024. Prediction might take a while, be less reliable."
                             "We advise not to run datasets > 10k samples, which might lead to your machine crashing (due to quadratic memory scaling of TabPFN)."
                             "Please confirm you want to run by passing overwrite_warning=True to the fit function.")

        # Return the classifier
        return self

    def predict_proba(self, X, normalize_with_test=False, return_logits=False):
        """
        Predict the probabilities for the input X depending on the training set previously passed in the method fit.

        If no_grad is true in the classifier the function takes X as a numpy.ndarray. If no_grad is false X must be a
        torch tensor and is not fully checked.
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        if self.no_grad:
            X = check_array(X, force_all_finite=False)
            X_full = np.concatenate([self.X_, X], axis=0)
            X_full = torch.tensor(X_full, device=self.device).float().unsqueeze(1)
        else:
            assert (torch.is_tensor(self.X_) & torch.is_tensor(X)), "If no_grad is false, this function expects X as " \
                                                                    "a tensor to calculate a gradient"
            X_full = torch.cat((self.X_, X), dim=0).float().unsqueeze(1).to(self.device)

            if int(torch.isnan(X_full).sum()):
                print('X contains nans and the gradient implementation is not designed to handel nans.')

        y_full = np.concatenate([self.y_, np.zeros(shape=X.shape[0])], axis=0)
        y_full = torch.tensor(y_full, device=self.device).float().unsqueeze(1)

        eval_pos = self.X_.shape[0]

        prediction = transformer_predict(self.model, X_full, y_full, eval_pos,
                                         device=self.device,
                                         style=self.style,
                                         inference_mode=True,
                                         preprocess_transform='none' if self.no_preprocess_mode else 'mix',
                                         normalize_with_test=normalize_with_test,
                                         N_ensemble_configurations=self.N_ensemble_configurations,
                                         softmax_temperature=self.temperature,
                                         multiclass_decoder=self.multiclass_decoder,
                                         feature_shift_decoder=self.feature_shift_decoder,
                                         differentiable_hps_as_style=self.differentiable_hps_as_style,
                                         seed=self.seed,
                                         return_logits=return_logits,
                                         no_grad=self.no_grad,
                                         batch_size_inference=self.batch_size_inference, scaling=self.scaling,
                                         **get_params_from_config(self.c))
        prediction_ = prediction.squeeze(0)

        return prediction_.detach().cpu().numpy() if self.no_grad else prediction_

    def predict(self, X, return_winning_probability=False, normalize_with_test=False):
        p = self.predict_proba(X, normalize_with_test=normalize_with_test)
        y = np.argmax(p, axis=-1)
        y = self.classes_.take(np.asarray(y, dtype=np.intp))
        if return_winning_probability:
            return y, p.max(axis=-1)
        return y


def predict(eval_xs, eval_ys, used_style, softmax_temperature, return_logits, model, eval_position, num_classes, inference_mode, no_grad):
    # Initialize results array size S, B, Classes
    # no_grad disables inference_mode, because otherwise the gradients are lost
    inference_mode_call = torch.inference_mode() if inference_mode and no_grad else NOP()
    with inference_mode_call:
        output = model(
            (used_style.repeat(eval_xs.shape[1], 1) if used_style is not None else None, eval_xs, eval_ys.float()),
            single_eval_pos=eval_position)[:, :, 0:num_classes]

        output = output[:, :, 0:num_classes] / torch.exp(softmax_temperature)
        if not return_logits:
            output = torch.nn.functional.softmax(output, dim=-1)

    return output


def preprocess_input(eval_xs, eval_ys, preprocess_transform, max_features, normalize_with_test, eval_position,
                     categorical_feats, device, normalize_to_ranking, normalize_with_sqrt, scaling):
    import warnings

    if eval_xs.shape[1] > 1:
        raise Exception("Transforms only allow one batch dim - TODO")

    if eval_xs.shape[2] > max_features:
        eval_xs = eval_xs[:, :, sorted(np.random.choice(eval_xs.shape[2], max_features, replace=False))]

    if preprocess_transform != 'none':
        if preprocess_transform == 'power' or preprocess_transform == 'power_all':
            pt = PowerTransformer(standardize=True)
        elif preprocess_transform == 'quantile' or preprocess_transform == 'quantile_all':
            pt = QuantileTransformer(output_distribution='normal')
        elif preprocess_transform == 'robust' or preprocess_transform == 'robust_all':
            pt = RobustScaler(unit_variance=True)

    if scaling:
        eval_xs = normalize_data(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position)
    else:
        eval_xs = torch.clip(eval_xs, min=-100, max=100)
        

    # Removing empty features
    eval_xs = eval_xs[:, 0, :]
    sel = [len(torch.unique(eval_xs[0:eval_ys.shape[0], col])) > 1 for col in range(eval_xs.shape[1])]
    # sel = eval_xs[0:eval_ys.shape[0]].var(dim=0) > 0
    eval_xs = eval_xs[:, sel]

    warnings.simplefilter('error')
    if preprocess_transform != 'none':
        eval_xs = eval_xs.cpu().numpy()
        feats = set(range(eval_xs.shape[1])) if 'all' in preprocess_transform else set(
            range(eval_xs.shape[1])) - set(categorical_feats)
        for col in feats:
            pt.fit(eval_xs[0:eval_position, col:col + 1])
            trans = pt.transform(eval_xs[:, col:col + 1])
            eval_xs[:, col:col + 1] = trans
        eval_xs = torch.tensor(eval_xs).float()
    warnings.simplefilter('default')

    eval_xs = eval_xs.unsqueeze(1)

    # TODO: Caution there is information leakage when to_ranking is used, we should not use it
    eval_xs = remove_outliers(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position) \
        if not normalize_to_ranking else normalize_data(to_ranking_low_mem(eval_xs))
    # Rescale X
    eval_xs = normalize_by_used_features_f(eval_xs, eval_xs.shape[-1], max_features,
                                            normalize_with_sqrt=normalize_with_sqrt)

    return eval_xs.to(device)


def transformer_predict(
        model, eval_xs, eval_ys, eval_position, device='cpu', max_features=100, style=None, inference_mode=False,
        num_classes=2, extend_features=True, normalize_with_test=False, normalize_to_ranking=False, softmax_temperature=0.0,
        multiclass_decoder='permutation', preprocess_transform='mix', categorical_feats=[], feature_shift_decoder=False,
        N_ensemble_configurations=10, batch_size_inference=16, differentiable_hps_as_style=False, average_logits=True,
        fp16_inference=False, normalize_with_sqrt=False, seed=0, no_grad=True, return_logits=False, scaling=True, **kwargs):
    
    num_classes = len(torch.unique(eval_ys))

    eval_xs, eval_ys = eval_xs.to(device), eval_ys.to(device)
    eval_ys = eval_ys[:eval_position]

    model.to(device)

    model.eval()

    import itertools
    if not differentiable_hps_as_style:
        style = None

    num_styles = 1
    style = None
    softmax_temperature = torch.log(torch.tensor([0.8]))

    styles_configurations = range(0, num_styles)

    preprocess_transform_configurations = ['none', 'power_all'] if preprocess_transform == 'mix' else [preprocess_transform]

    if seed is not None:
        torch.manual_seed(seed)

    # we don't really need to do a permutation if we're gonna shuffle, right?
    feature_shift_configurations = torch.randperm(eval_xs.shape[2]) if feature_shift_decoder else [0]
    class_shift_configurations = torch.randperm(len(torch.unique(eval_ys))) if multiclass_decoder == 'permutation' else [0]

    ensemble_configurations = list(itertools.product(class_shift_configurations, feature_shift_configurations))

    rng = random.Random(seed)
    rng.shuffle(ensemble_configurations)
    ensemble_configurations = list(itertools.product(ensemble_configurations, preprocess_transform_configurations, styles_configurations))
    ensemble_configurations = ensemble_configurations[0:N_ensemble_configurations]

    output = None
    eval_xs_transformed = {}
    inputs, labels = [], []
    for ensemble_configuration in ensemble_configurations:
        (class_shift_configuration, feature_shift_configuration), preprocess_transform_configuration, styles_configuration = ensemble_configuration

        style_ = style[styles_configuration:styles_configuration+1, :] if style is not None else style
        softmax_temperature_ = softmax_temperature[styles_configuration]

        eval_xs_, eval_ys_ = eval_xs.clone(), eval_ys.clone()

        if preprocess_transform_configuration in eval_xs_transformed:
            eval_xs_ = eval_xs_transformed[preprocess_transform_configuration].clone()
        else:
            eval_xs_ = preprocess_input(eval_xs_, eval_ys, preprocess_transform=preprocess_transform_configuration, max_features=max_features,
                                        normalize_with_test=normalize_with_test, eval_position=eval_position, categorical_feats=categorical_feats,
                                        device=device, normalize_to_ranking=normalize_to_ranking, normalize_with_sqrt=normalize_with_sqrt,
                                        scaling=scaling)
            if no_grad:
                eval_xs_ = eval_xs_.detach()
            eval_xs_transformed[preprocess_transform_configuration] = eval_xs_

        eval_ys_ = ((eval_ys_ + class_shift_configuration) % num_classes).float()

        eval_xs_ = torch.cat([eval_xs_[..., feature_shift_configuration:], eval_xs_[..., :feature_shift_configuration]], dim=-1)

        # Extend X
        if extend_features:
            eval_xs_ = torch.cat(
                [eval_xs_,
                 torch.zeros((eval_xs_.shape[0], eval_xs_.shape[1], max_features - eval_xs_.shape[2])).to(device)], -1)
        inputs += [eval_xs_]
        labels += [eval_ys_]

    inputs = torch.cat(inputs, 1)
    inputs = torch.split(inputs, batch_size_inference, dim=1)
    labels = torch.cat(labels, 1)
    labels = torch.split(labels, batch_size_inference, dim=1)
    outputs = []
    for batch_input, batch_label in zip(inputs, labels):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message="None of the inputs have requires_grad=True. Gradients will be None")
            warnings.filterwarnings("ignore",
                                    message="torch.cuda.amp.autocast only affects CUDA ops, but CUDA is not available.  Disabling.")
            if device == 'cpu':
                output_batch = checkpoint(predict, batch_input, batch_label, style_, softmax_temperature_, True,  model, eval_position, num_classes, inference_mode, no_grad)

            else:
                with torch.cuda.amp.autocast(enabled=fp16_inference):
                    output_batch = checkpoint(predict, batch_input, batch_label, style_, softmax_temperature_, True, model, eval_position, num_classes, inference_mode, no_grad)
        outputs += [output_batch]

    outputs = torch.cat(outputs, 1)
    for i, ensemble_configuration in enumerate(ensemble_configurations):
        (class_shift_configuration, feature_shift_configuration), preprocess_transform_configuration, styles_configuration = ensemble_configuration
        output_ = outputs[:, i:i+1, :]
        output_ = torch.cat([output_[..., class_shift_configuration:], output_[..., :class_shift_configuration]], dim=-1)

        if not average_logits and not return_logits:
            # transforms every ensemble_configuration into a probability -> equal contribution of every configuration
            output_ = torch.nn.functional.softmax(output_, dim=-1)
        output = output_ if output is None else output + output_

    output = output / len(ensemble_configurations)
    if average_logits and not return_logits:
        output = torch.nn.functional.softmax(output, dim=-1)

    output = torch.transpose(output, 0, 1)

    return output


def get_params_from_config(c):
    return {'max_features': c['num_features'], 'rescale_features': c["normalize_by_used_features"], 'normalize_to_ranking': c["normalize_to_ranking"], 'normalize_with_sqrt': c.get("normalize_with_sqrt", False)
            }
