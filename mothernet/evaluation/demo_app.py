from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

from bokeh.io import show
from bokeh.models import CustomJS, Dropdown

from bokeh.layouts import gridplot, layout
from bokeh.plotting import figure, show
from bokeh.plotting import curdoc
import time

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from mothernet.prediction.mothernet_additive import MotherNetAdditiveClassifier

import torch
from sklearn.feature_selection import f_classif, SelectKBest, VarianceThreshold
from sklearn.metrics import roc_auc_score
torch.set_num_threads(1)
from sklearn.pipeline import make_pipeline
import numpy as np
from bokeh.models import Div, CheckboxButtonGroup

from mothernet.utils import get_mn_model

grid_figures = {}


def plot_shape_function(bin_edges: np.ndarray, w: np.ndarray, feature_names=None, feature_subset=None):
    num_features = len(feature_subset) if feature_subset is not None else len(bin_edges)
    columns = min(int(np.ceil(np.sqrt(num_features))), 6)
    rows = int(np.ceil(num_features / columns))
    feature_range = feature_subset if feature_subset is not None else range(num_features)
    figures = []
    for ax_idx, feature_idx in enumerate(feature_range):
        weights_normalized = w[feature_idx][0:-1][:, 1] - w[feature_idx][0:-1].mean(axis=-1)
        if feature_names is None:
            title = f'Feature {feature_idx}'
        else:
            title = f'{feature_names[feature_idx]}'
        p = figure(width=200, height=100, title=title)
        my_step = p.step(bin_edges[feature_idx], weights_normalized)
        grid_figures[title] = (p, my_step)
        figures.append(p)
    grid = gridplot(zip(*([iter(figures)] * columns)), width=250, height=250, toolbar_location=None)
    return grid



columns = (['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot'
,'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations'
,'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate'
,'srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count'
,'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate'
,'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','attack','level'])
df_train = pd.read_csv("https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B_20Percent.txt", names=columns, header=None)

df_train['is_attack'] = df_train.attack != "normal"

X_train = df_train.drop(columns=["attack", "level", 'is_attack'])
y_train = df_train['is_attack']

df_test = pd.read_csv("https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt", names=columns, header=None)
X_test = df_test.drop(columns=["attack", "level"])
y_test = df_test['attack'] != "normal"


drop_cols = []
for col in X_train.columns:
    if (X_train[col] == X_train[col].mode()[0]).mean() > .85:
        drop_cols.append(col)
X_train = X_train.drop(columns=drop_cols)

cat_cols = X_train.dtypes.index[X_train.dtypes == "object"]
cont_cols = X_train.dtypes.index[X_train.dtypes != "object"]


def fit_predict_gamma_net(filter_feature=None, filter_value=None):
    device = "cpu"
    #additive = MotherNetAdditiveClassifier(path="../models_diff/baam_H512_Dclass_average_e128_nsamples500_numfeatures20_padzerosFalse_03_14_2024_15_03_22_epoch_1520.cpkt", device=device)
    ct = make_column_transformer((OneHotEncoder(sparse_output=False, max_categories=10, handle_unknown='ignore'), X_train.dtypes == object), remainder="passthrough", verbose_feature_names_out=False)

    model_string = "baam_H512_Dclass_average_e128_nsamples500_numfeatures20_padzerosFalse_03_14_2024_15_03_22_epoch_1520.cpkt"
    model_path = get_mn_model(model_string)

    additive = MotherNetAdditiveClassifier(path=model_path, device="cpu")
    pipe = make_pipeline(ct, VarianceThreshold(), additive)

    subsample = np.random.permutation(X_train.shape[0])[:100]
    vals.label = value_item
    if filter_value == "None":
        mask = np.ones(X_train.shape[0], dtype="bool")
        mask_test = np.ones(X_test.shape[0], dtype="bool")
    else:
        mask = X_train[filter_feature] == value_item
        mask_test = X_test[filter_feature] == value_item
    some_output.text = "fitting..."
    X_train_masked = X_train[mask]
    y_train_masked = y_train[mask]
    if len(uniques := np.unique(y_train_masked)) == 1:
        some_output.text = f"only one class: {uniques[0]}"
        return
    success = False
    while not success:
        subsample = np.random.permutation(X_train_masked.shape[0])[:100]
        if y_train_masked.iloc[subsample].nunique() > 1:
            success = True
    tick = time.time()
    pipe.fit(X_train_masked.iloc[subsample], y_train_masked.iloc[subsample])
    fitting_time = time.time() - tick
    tick = time.time()
    X_test_masked = X_test[mask_test]
    y_test_masked = y_test[mask_test]
    auc = roc_auc_score(y_test_masked, pipe.predict_proba(X_test_masked)[:, 1])
    scoring_time = time.time() - tick
    some_output.text = f"fitting time: {fitting_time:.2f}s, AUC: {auc:.2f}"
    print(f"fitting time: {fitting_time:.2f}s")
    feature_names = pipe[:-1].get_feature_names_out()
    selected_features = [list(feature_names).index(col) for col in cont_cols if col in feature_names]
    additive = pipe[-1]
    return pipe, additive.bin_edges_, additive.w_, feature_names, selected_features

# axes = plot_shape_function(additive.bin_edges_, additive.w_, feature_names=feature_names, feature_subset=selected_features)

select = ["None"] + list(cat_cols)
per_col_cats = X_train[cat_cols].apply(lambda x: pd.unique(x).tolist()).to_dict()


menu = [(s, s) for s in select]

cats = Dropdown(label="None", menu=menu, width=100)
vals = Dropdown(label="None", menu=[("None", "None")], width=100)

slice_label = Div(text="Slice By", margin=(10, 2, 0, 20))
value_label = Div(text="Value", margin=(10, 2, 0, 20))
some_output = Div(text="", margin=(10, 2, 0, 20))

checkbox_button_group = CheckboxButtonGroup(labels=['EBM', 'GammaNet'], active=[])


def pick_feature(event):
    print("callback")
    print(event)
    selection = event.item
    print(selection)
    cats.label = selection
    if selection != "None":
        old_vals = vals.label
        new_menu = [("None", "None")] + [(str(v), str(v)) for v in per_col_cats[selection]]
        print(vals.menu)
        vals.update(label="None", menu= new_menu)
        vals.label = "bla"
        vals.label = "None"
        if old_vals != "None":
            fit_predict_with_model()
    else:
        vals.menu = [("None", "None")]
        vals.label = "None"
        fit_predict_with_model()

def select_val(event):
    print("select val callback")
    print(event)
    value_item = event.item
    print(f"selected value: {value_item}")
    fit_predict_with_model()

def update_plot(bin_edges, w, feature_names):
    for bins, w, feature in zip(bin_edges, w, feature_names):
        if feature not in grid_figures:
            continue
        p, my_step = grid_figures[feature]
        weights_normalized = w[0:-1][:, 1] - w[0:-1].mean(axis=-1)
        my_step.data_source.data['y'] = weights_normalized
        my_step.data_source.data['x'] = bins

def get_model():
    if 0 in checkbox_button_group.active:
        return "ebm"
    if 1 in checkbox_button_group.active:
        return "gamma_net"
    return None

def fit_predict_with_model():
    model = get_model()
    if model == "ebm":
        print("fitting ebm")
        return
    if model == "gamma_net":
        print("fitting gamma_net")
        pipe, bin_edges, w, feature_names, selected_features = fit_predict_gamma_net(cats.label, vals.label)

    if len(grid_figures) == 0:
        plot_shape_function(bin_edges, w, feature_names=feature_names, feature_subset=selected_features)
    else:
        update_plot(bin_edges, w, feature_names)
    


def checkbox_event(attr, old, new):
    if len(new) == 2:
        new_val = [x for x in new if x not in old]
        print(f"setting to {new_val}")
        checkbox_button_group.active = new_val
    if new_val != old[0]:
        fit_predict_with_model()
                

#cats.on_change('label', my_func)
cats.on_click(pick_feature)
vals.on_click(select_val)
checkbox_button_group.on_change("active", checkbox_event)
col = layout([[checkbox_button_group, slice_label, cats, value_label, vals, some_output], [axes]])
print("prestart")
curdoc().add_root(col)
print("starting")
