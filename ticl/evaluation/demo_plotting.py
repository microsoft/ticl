import ipywidgets as widgets
import time
import io

with open("plot_test.png", 'rb') as file:
    image = file.read()
    
img = widgets.Image(
    value=image,
    format='png',
    width=1600,
    height=1800,
)

cats = widgets.Dropdown(
    options=select,
    value='None',
    description='Slice By',
    disabled=False,
)

vals = widgets.Dropdown(
    options=["None"],
    value='None',
    description='Filter',
    disabled=False,
)

label = widgets.Label(value="")
out = widgets.Output()


def plot_with_filter(column, value):
    if value == "None":
        mask = np.ones(X_train.shape[0], dtype="bool")
    else:
        mask = X_train[column] == value
    label.value = "fitting..."
    X_train_masked = X_train[mask]
    y_train_masked = y_train[mask]
    subsample = np.random.permutation(X_train_masked.shape[0])[:1000]
    tick = time.time()
    pipe.fit(X_train_masked.iloc[subsample], y_train_masked.iloc[subsample])
    tock = time.time() - tick
    label.value = f"fitting time: {tock:.2f}s"
    print("finished fitting")
    feature_names = pipe[:-1].get_feature_names_out()
    selected_features = [list(feature_names).index(col) for col in cont_cols if col in feature_names]
    additive = pipe[-1]
    axes = plot_shape_function(additive.bin_edges_, additive.w_, feature_names=feature_names, feature_subset=selected_features)
    memfile = io.BytesIO()
    plt.savefig(memfile, bbox_inches="tight")
    memfile.seek(0)
    image = memfile.read()
    img.value = image

def set_cat_values(change):
    if change.new == "None":
        vals.options = ["None"]
    else:
        vals.options = ["None"] + per_col_cats[change.new]
        vals.value = "None"

def plot_with_vals(change):
    plot_with_filter(cats.value, vals.value)

cats.observe(set_cat_values, names="value")
vals.observe(plot_with_vals, names="value")
widgets.VBox([widgets.HBox([cats, vals, label]),
             img])

