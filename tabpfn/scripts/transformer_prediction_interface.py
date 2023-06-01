import torch
import random
import pathlib

from torch.utils.checkpoint import checkpoint

from tabpfn.utils import normalize_data, to_ranking_low_mem, remove_outliers
from tabpfn.utils import NOP, normalize_by_used_features_f

from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import column_or_1d
from sklearn.utils import gen_batches
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from pathlib import Path
from tabpfn.scripts.model_builder import load_model, load_model_only_inference
import os
import pickle
import io
from tqdm import tqdm
from annoy import AnnoyIndex

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'Manager':
            from settings import Manager
            return Manager
        try:
            return self.find_class_cpu(module, name)
        except:
            return None

    def find_class_cpu(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def load_model_workflow(e, add_name, base_path, device='cpu', eval_addition='', only_inference=True, verbose=0):
    """
    Workflow for loading a model and setting appropriate parameters for diffable hparam tuning.

    :param e:
    :param eval_positions_valid:
    :param add_name:
    :param base_path:
    :param device:
    :param eval_addition:
    :return:
    """
    def get_file(e):
        """
        Returns the different paths of model_file, model_path and results_file
        """
        model_file = f'models_diff/{add_name}_epoch_{e}.cpkt'
        model_path = os.path.join(base_path, model_file)
        # print('Evaluate ', model_path)
        results_file = os.path.join(base_path,
                                    f'models_diff/prior_diff_real_results_{add_name}_n_0_epoch_{e}_{eval_addition}.pkl')
        return model_file, model_path, results_file

    def check_file(e):
        model_file, model_path, results_file = get_file(e)
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
                model_file = None
        else:
            if verbose:
                print(f"loading model from file {model_file}")
        return model_file, model_path, results_file

    model_file = None
    if e == -1:
        for e_ in range(100, -1, -1):
            model_file_, model_path_, results_file_ = check_file(e_)
            if model_file_ is not None:
                e = e_
                model_file, model_path, results_file = model_file_, model_path_, results_file_
                break
    else:
        model_file, model_path, results_file = check_file(e)

    if model_file is None:
        model_file, model_path, results_file = get_file(e)
        raise Exception('No checkpoint found at '+str(model_path))


    #print(f'Loading {model_file}')
    if only_inference:
        if verbose:
            print('Loading model that can be used for inference only')
        model, c = load_model_only_inference(base_path, model_file, device)
    else:
        #until now also only capable of inference
        model, c = load_model(base_path, model_file, device, eval_positions=[], verbose=False)
    #model, c = load_model(base_path, model_file, device, eval_positions=[], verbose=False)

    return model, c, results_file


class TabPFNClassifier(BaseEstimator, ClassifierMixin):

    models_in_memory = {}

    def __init__(self, device='cpu', epoch=-1, base_path=pathlib.Path(__file__).parent.parent.resolve(), model_string='download',
                 N_ensemble_configurations=3, combine_preprocessing=False, no_preprocess_mode=False,
                 multiclass_decoder='permutation', feature_shift_decoder=True, only_inference=True, seed=0, verbose=0, temperature=1, model=None):
        # Model file specification (Model name, Epoch)
        self.epoch = epoch
        model_key = model_string+'|'+str(device)
        if model_string in self.models_in_memory:
            if verbose:
                print(f"using model {model_key}")
            model, c, results_file = self.models_in_memory[model_key]
        else:
            model, c, results_file = load_model_workflow(epoch, add_name=model_string, base_path=base_path, device=device,
                                                         eval_addition='', only_inference=only_inference)
            self.models_in_memory[model_key] = (model, c, results_file)
            if len(self.models_in_memory) == 2:
                print('Multiple models in memory. This might lead to memory issues. Consider calling remove_models_from_memory()')
        #style, temperature = self.load_result_minimal(style_file, i, e)

        self.verbose = verbose
        self.device = device
        self.model = model
        self.c = c
        self.style = None
        self.temperature = temperature
        self.N_ensemble_configurations = N_ensemble_configurations
        self.base__path = base_path
        self.base_path = base_path
        self.model_string = model_string

        self.max_num_features = self.c['num_features']
        self.max_num_classes = self.c['max_num_classes']
        self.differentiable_hps_as_style = self.c['differentiable_hps_as_style']

        self.no_preprocess_mode = no_preprocess_mode
        self.combine_preprocessing = combine_preprocessing
        self.feature_shift_decoder = feature_shift_decoder
        self.multiclass_decoder = multiclass_decoder
        self.only_inference = only_inference
        self.seed = seed

    def remove_models_from_memory(self):
        self.models_in_memory = {}

    def load_result_minimal(self, path, i, e):
        with open(path, 'rb') as output:
            _, _, _, style, temperature, optimization_route = CustomUnpickler(output).load()

            return style, temperature

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
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, force_all_finite=False)
        # Store the classes seen during fit
        y = self._validate_targets(y)
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)

        self.X_ = X
        self.y_ = y

        if X.shape[1] > self.max_num_features:
            raise ValueError("The number of features for this classifier is restricted to ", self.max_num_features)
        if len(np.unique(y)) > self.max_num_classes:
            raise ValueError("The number of classes for this classifier is restricted to ", self.max_num_classes)
        if X.shape[0] > 1024 and not overwrite_warning:
            raise ValueError("⚠️ WARNING: TabPFN is not made for datasets with a trainingsize > 1024. Prediction might take a while, be less reliable. We advise not to run datasets > 10k samples, which might lead to your machine crashing (due to quadratic memory scaling of TabPFN). Please confirm you want to run by passing overwrite_warning=True to the fit function.")
            

        # Return the classifier
        return self

    def predict_proba(self, X, normalize_with_test=False):
        batches = gen_batches(len(X), batch_size=1024)
        probas = []
        for batch in batches:
            probas.append(self._predict_proba(X[batch], normalize_with_test=normalize_with_test))
        probas = np.concatenate(probas, axis=0)
        return probas

    def _predict_proba(self, X, normalize_with_test=False):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X, force_all_finite=False)
        X_full = np.concatenate([self.X_, X], axis=0)
        X_full = torch.tensor(X_full, device=self.device).float().unsqueeze(1)
        y_full = np.concatenate([self.y_, np.zeros_like(X[:, 0])], axis=0)
        y_full = torch.tensor(y_full, device=self.device).float().unsqueeze(1)

        eval_pos = self.X_.shape[0]

        prediction = transformer_predict(self.model[2], X_full, y_full, eval_pos,
                                         device=self.device,
                                         style=self.style,
                                         inference_mode=True,
                                         preprocess_transform='none' if self.no_preprocess_mode else 'mix',
                                         normalize_with_test=normalize_with_test,
                                         N_ensemble_configurations=self.N_ensemble_configurations,
                                         softmax_temperature=np.log(self.temperature),
                                         combine_preprocessing=self.combine_preprocessing,
                                         multiclass_decoder=self.multiclass_decoder,
                                         feature_shift_decoder=self.feature_shift_decoder,
                                         differentiable_hps_as_style=self.differentiable_hps_as_style,
                                         seed=self.seed, verbose=self.verbose,
                                         **get_params_from_config(self.c))
        prediction_, y_ = prediction.squeeze(0), y_full.squeeze(1).long()[eval_pos:]

        return prediction_.detach().cpu().numpy()

    def predict(self, X, return_winning_probability=False, normalize_with_test=False):
        p = self.predict_proba(X, normalize_with_test=normalize_with_test)
        y = np.argmax(p, axis=-1)
        y = self.classes_.take(np.asarray(y, dtype=np.intp))
        if return_winning_probability:
            return y, p.max(axis=-1)
        return y

class ApproxNNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_neighbors=10, n_trees=10):
        self.n_neighbors = n_neighbors
        self.n_trees = n_trees

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.X_ = np.array(X)
        self.y_ = np.array(y)
        self.ann_index = AnnoyIndex(self.X_.shape[1], 'angular')
        for i in range(len(X)):
            v = self.X_[i]
            self.ann_index.add_item(i, v)
        self.ann_index.build(self.n_trees)
        try: 
            self.y_onehot_  = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
        except TypeError:  # old sklearn
            self.y_onehot_  = OneHotEncoder(sparse=False).fit_transform(y.reshape(-1, 1))
        return self

    def predict_proba(self, X):
        X = np.array(X)
        pred_probs = []
        for i in range(len(X)):
            closest = self.ann_index.get_nns_by_vector(X[i], self.n_neighbors)
            this_y_train = self.y_onehot_[closest]
            pred_probs.append(this_y_train.mean(axis=0))
        y_pred_prob = np.c_[pred_probs]
        return y_pred_prob
    
    def predict(self, X):
        p = self.predict_proba(X)
        y = np.argmax(p, axis=-1)
        return self.classes_.take(np.asarray(y, dtype=np.intp))

class NeighborsMetaClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, clf=None, predict_batch_size=10, n_neighbors=10, n_trees_annoy=10, verbose=0, overwrite_warning=False):
        self.predict_batch_size = predict_batch_size
        self.n_neighbors = n_neighbors
        if clf is None:
            clf = TabPFNClassifier(**self.kwargs)
        self.clf = clf
        self.verbose = verbose
        self.overwrite_warning = overwrite_warning
        self.n_trees_annoy = n_trees_annoy
    
    def fit(self, X, y):
        self.X_ = np.array(X)
        self.y_ = np.array(y)
        self.classes_ = np.unique(self.y_)
        self.le_ = LabelEncoder().fit(self.y_)

        f = self.X_.shape[1]
        self.ann_index = AnnoyIndex(f, 'angular')
        for i in range(len(X)):
            v = self.X_[i]
            self.ann_index.add_item(i, v)
        self.ann_index.build(self.n_trees_annoy)
        return self

    def predict_proba(self, X):
        X = np.array(X)
        pred_probs = []
        batches = gen_batches(len(X), batch_size=self.predict_batch_size)
        if self.verbose :
            iter = tqdm(list(batches))
        else:
            iter = batches

        for batch in iter:
            closest = []
            this_X_test = X[batch]
            for row in this_X_test:
                closest.extend(self.ann_index.get_nns_by_vector(row, self.n_neighbors))
            closest = np.unique(closest)
            
            this_X_train = self.X_[closest]
            this_y_train = self.y_[closest]
            if len(np.unique(this_y_train)) > 1:
                if isinstance(self.clf, TabPFNClassifier):
                    self.clf.fit(this_X_train, this_y_train, overwrite_warning=self.overwrite_warning)
                else:
                    self.clf.fit(this_X_train, this_y_train)

                y_pred_prob = self.clf.predict_proba(this_X_test)
                if y_pred_prob.shape[1] != len(self.classes_):
                    reshaped_y_prob = np.zeros((len(this_X_test), len(self.classes_)))
                    unique_y_indexes = self.le_.transform(np.unique(this_y_train))
                    reshaped_y_prob[:, unique_y_indexes] = y_pred_prob
                    y_pred_prob = reshaped_y_prob
            else:
                # only one class present, gotta be that class
                try:
                    y_pred_prob = OneHotEncoder(sparse_output=False, categories=[self.classes_]).fit_transform(this_y_train[[0]].reshape(-1, 1)).repeat(len(this_X_test), axis=0)
                except TypeError:
                    y_pred_prob = OneHotEncoder(sparse=False, categories=[self.classes_]).fit_transform(this_y_train[[0]].reshape(-1, 1)).repeat(len(this_X_test), axis=0)

            assert len(y_pred_prob) == len(this_X_test)
            pred_probs.append(y_pred_prob)
        results = np.concatenate(pred_probs)
        assert len(results) == len(X)
        return results

    def predict(self, X):
        p = self.predict_proba(X)
        y = np.argmax(p, axis=-1)
        return self.classes_.take(np.asarray(y, dtype=np.intp))


import time
def transformer_predict(model, eval_xs, eval_ys, eval_position,
                        device='cpu',
                        max_features=100,
                        style=None,
                        inference_mode=False,
                        num_classes=2,
                        extend_features=True,
                        normalize_with_test=False,
                        normalize_to_ranking=False,
                        softmax_temperature=0.0,
                        multiclass_decoder='permutation',
                        preprocess_transform='mix',
                        categorical_feats=[],
                        feature_shift_decoder=False,
                        N_ensemble_configurations=10,
                        combine_preprocessing=False,
                        batch_size_inference=16,
                        differentiable_hps_as_style=False,
                        average_logits=True,
                        fp16_inference=False,
                        normalize_with_sqrt=False,
                        seed=0,
                        **kwargs):
    """

    :param model:
    :param eval_xs:
    :param eval_ys:
    :param eval_position:
    :param rescale_features:
    :param device:
    :param max_features:
    :param style:
    :param inference_mode:
    :param num_classes:
    :param extend_features:
    :param normalize_to_ranking:
    :param softmax_temperature:
    :param multiclass_decoder:
    :param preprocess_transform:
    :param categorical_feats:
    :param feature_shift_decoder:
    :param N_ensemble_configurations:
    :param average_logits:
    :param normalize_with_sqrt:
    :param metric_used:
    :return:
    """
    num_classes = len(torch.unique(eval_ys))

    def predict(eval_xs, eval_ys, used_style, softmax_temperature, return_logits):
        # Initialize results array size S, B, Classes

        inference_mode_call = torch.inference_mode() if inference_mode else NOP()
        with inference_mode_call:
            start = time.time()
            output = model(
                    (used_style.repeat(eval_xs.shape[1], 1) if used_style is not None else None, eval_xs, eval_ys.float()),
                    single_eval_pos=eval_position)[:, :, 0:num_classes]

            output = output[:, :, 0:num_classes] / torch.exp(softmax_temperature)
            if not return_logits:
                output = torch.nn.functional.softmax(output, dim=-1)
            #else:
            #    output[:, :, 1] = model((style.repeat(eval_xs.shape[1], 1) if style is not None else None, eval_xs, eval_ys.float()),
            #               single_eval_pos=eval_position)

            #    output[:, :, 1] = torch.sigmoid(output[:, :, 1]).squeeze(-1)
            #    output[:, :, 0] = 1 - output[:, :, 1]

        #print('RESULTS', eval_ys.shape, torch.unique(eval_ys, return_counts=True), output.mean(axis=0))

        return output

    def preprocess_input(eval_xs, preprocess_transform):
        import warnings

        if eval_xs.shape[1] > 1:
            raise Exception("Transforms only allow one batch dim - TODO")
        if preprocess_transform != 'none':
            if preprocess_transform == 'power' or preprocess_transform == 'power_all':
                pt = PowerTransformer(standardize=True)
            elif preprocess_transform == 'quantile' or preprocess_transform == 'quantile_all':
                pt = QuantileTransformer(output_distribution='normal')
            elif preprocess_transform == 'robust' or preprocess_transform == 'robust_all':
                pt = RobustScaler(unit_variance=True)

        # eval_xs, eval_ys = normalize_data(eval_xs), normalize_data(eval_ys)
        eval_xs = normalize_data(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position)

        # Removing empty features
        eval_xs = eval_xs[:, 0, :]
        sel = eval_xs[0:eval_ys.shape[0]].var(dim=0) > 0
        # sel2 = [len(torch.unique(eval_xs[0:eval_ys.shape[0], col])) > 1 for col in range(eval_xs.shape[1])]
        # if (np.array(sel) != np.array(sel2)).any():
        #    import pdb; pdb.set_trace()
        eval_xs = eval_xs[:, sel]

        warnings.simplefilter('error')
        if preprocess_transform != 'none':
            eval_xs = eval_xs.cpu().numpy()
            feats = set(range(eval_xs.shape[1])) if 'all' in preprocess_transform else set(
                range(eval_xs.shape[1])) - set(categorical_feats)
            for col in feats:
                try:
                    pt.fit(eval_xs[0:eval_position, col:col + 1])
                    trans = pt.transform(eval_xs[:, col:col + 1])
                    # print(scipy.stats.spearmanr(trans[~np.isnan(eval_xs[:, col:col+1])], eval_xs[:, col:col+1][~np.isnan(eval_xs[:, col:col+1])]))
                    eval_xs[:, col:col + 1] = trans
                except:
                    pass
            eval_xs = torch.tensor(eval_xs).float()
        warnings.simplefilter('default')

        eval_xs = eval_xs.unsqueeze(1)

        # TODO: Cautian there is information leakage when to_ranking is used, we should not use it
        eval_xs = remove_outliers(eval_xs, normalize_positions=-1 if normalize_with_test else eval_position) if not normalize_to_ranking else normalize_data(to_ranking_low_mem(eval_xs))
        # Rescale X
        eval_xs = normalize_by_used_features_f(eval_xs, eval_xs.shape[-1], max_features,
                                               normalize_with_sqrt=normalize_with_sqrt)

        return eval_xs.detach().to(device)

    eval_xs, eval_ys = eval_xs.to(device), eval_ys.to(device)
    eval_ys = eval_ys[:eval_position]

    model.to(device)

    model.eval()

    import itertools
    if not differentiable_hps_as_style:
        style = None

    if style is not None:
        style = style.to(device)
        style = style.unsqueeze(0) if len(style.shape) == 1 else style
        num_styles = style.shape[0]
        softmax_temperature = softmax_temperature if softmax_temperature.shape else softmax_temperature.unsqueeze(
            0).repeat(num_styles)
    else:
        num_styles = 1
        style = None
        softmax_temperature = torch.log(torch.tensor([0.8]))

    styles_configurations = range(0, num_styles)
    def get_preprocess(i):
        if i == 0:
            return 'power_all'
#            if i == 1:
#                return 'robust_all'
        if i == 1:
            return 'none'

    preprocess_transform_configurations = ['none', 'power_all'] if preprocess_transform == 'mix' else [preprocess_transform]

    if seed is not None:
        torch.manual_seed(seed)

    feature_shift_configurations = torch.randperm(eval_xs.shape[2]) if feature_shift_decoder else [0]
    class_shift_configurations = torch.randperm(len(torch.unique(eval_ys))) if multiclass_decoder == 'permutation' else [0]

    ensemble_configurations = list(itertools.product(class_shift_configurations, feature_shift_configurations))
    #default_ensemble_config = ensemble_configurations[0]

    rng = random.Random(seed)
    rng.shuffle(ensemble_configurations)
    ensemble_configurations = list(itertools.product(ensemble_configurations, preprocess_transform_configurations, styles_configurations))
    ensemble_configurations = ensemble_configurations[0:N_ensemble_configurations]
    #if N_ensemble_configurations == 1:
    #    ensemble_configurations = [default_ensemble_config]

    output = None

    eval_xs_transformed = {}
    inputs, labels = [], []
    start = time.time()
    for ensemble_configuration in ensemble_configurations:
        (class_shift_configuration, feature_shift_configuration), preprocess_transform_configuration, styles_configuration = ensemble_configuration

        style_ = style[styles_configuration:styles_configuration+1, :] if style is not None else style
        softmax_temperature_ = softmax_temperature[styles_configuration]

        eval_xs_, eval_ys_ = eval_xs.clone(), eval_ys.clone()

        if preprocess_transform_configuration in eval_xs_transformed:
            eval_xs_ = eval_xs_transformed[preprocess_transform_configuration].clone()
        else:
            if eval_xs_.shape[-1] * 3 < max_features and combine_preprocessing:
                eval_xs_ = torch.cat([preprocess_input(eval_xs_, preprocess_transform='power_all'),
                            preprocess_input(eval_xs_, preprocess_transform='quantile_all')], -1)
                eval_xs_ = normalize_data(eval_xs_, normalize_positions=-1 if normalize_with_test else eval_position)
                #eval_xs_ = torch.stack([preprocess_input(eval_xs_, preprocess_transform='power_all'),
                #                        preprocess_input(eval_xs_, preprocess_transform='robust_all'),
                #                        preprocess_input(eval_xs_, preprocess_transform='none')], -1)
                #eval_xs_ = torch.flatten(torch.swapaxes(eval_xs_, -2, -1), -2)
            else:
                eval_xs_ = preprocess_input(eval_xs_, preprocess_transform=preprocess_transform_configuration)
            eval_xs_transformed[preprocess_transform_configuration] = eval_xs_

        eval_ys_ = ((eval_ys_ + class_shift_configuration) % num_classes).float()

        eval_xs_ = torch.cat([eval_xs_[..., feature_shift_configuration:],eval_xs_[..., :feature_shift_configuration]],dim=-1)

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
    #print('PREPROCESSING TIME', str(time.time() - start))
    outputs = []
    start = time.time()
    for batch_input, batch_label in zip(inputs, labels):
        #preprocess_transform_ = preprocess_transform if styles_configuration % 2 == 0 else 'none'
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message="None of the inputs have requires_grad=True. Gradients will be None")
            warnings.filterwarnings("ignore",
                                    message="torch.cuda.amp.autocast only affects CUDA ops, but CUDA is not available.  Disabling.")
            if device == 'cpu':
                output_batch = checkpoint(predict, batch_input, batch_label, style_, softmax_temperature_, True)
            else:
                with torch.cuda.amp.autocast(enabled=fp16_inference):
                    output_batch = checkpoint(predict, batch_input, batch_label, style_, softmax_temperature_, True)
        outputs += [output_batch]
    #print('MODEL INFERENCE TIME ('+str(batch_input.device)+' vs '+device+', '+str(fp16_inference)+')', str(time.time()-start))

    outputs = torch.cat(outputs, 1)
    for i, ensemble_configuration in enumerate(ensemble_configurations):
        (class_shift_configuration, feature_shift_configuration), preprocess_transform_configuration, styles_configuration = ensemble_configuration
        output_ = outputs[:, i:i+1, :]
        output_ = torch.cat([output_[..., class_shift_configuration:],output_[..., :class_shift_configuration]],dim=-1)

        #output_ = predict(eval_xs, eval_ys, style_, preprocess_transform_)
        if not average_logits:
            output_ = torch.nn.functional.softmax(output_, dim=-1)
        output = output_ if output is None else output + output_

    output = output / len(ensemble_configurations)
    if average_logits:
        output = torch.nn.functional.softmax(output, dim=-1)

    output = torch.transpose(output, 0, 1)

    return output

def get_params_from_config(c):
    return {'max_features': c['num_features']
        , 'rescale_features': c["normalize_by_used_features"]
        , 'normalize_to_ranking': c["normalize_to_ranking"]
        , 'normalize_with_sqrt': c.get("normalize_with_sqrt", False)
            }
