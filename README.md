# ticl - Tabular In-Context Learning

This repository contains code for training and prediction of several models for tabular in-context learning, including MotherNet and GAMformer.
MotherNet is a hypernetwork foundational model (or conditional neural process) for tabular data classification that creates a small neural network.
GAMformer is a model trained to output an interpretable, additive model using in-context learning.

Both the architecture and the code in this repository is based on the [TabPFN](https://github.com/automl/TabPFN) by the [Freiburg AutoML group](https://www.automl.org/).

This is a research prototype, shared for research use, and not meant for real-world applications. Responsibility for using the models contained in this repository,
as well monitoring and assessing potential impact of the models lies with the user of the code.

## Installation

It's recommended to use conda to create an environment using the provided environment file:

```
conda create -f environment.yml
```

## Getting started

A simple usage of our sklearn interface is:
```python
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from ticl.prediction import MotherNetClassifier, EnsembleMeta

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# MotherNetClassifier encapsulates a single instantiation of the model.

classifier = MotherNetClassifier(device='cpu', model_path="path/to/model.pkl")

classifier.fit(X_train, y_train)
y_eval, p_eval = classifier.predict(X_test, return_winning_probability=True)

print('Accuracy', accuracy_score(y_test, y_eval))

# Ensembling as described in the TabPFN paper an be performed using the EnsembleMeta wrapper
ensemble_classifier = EnsembleMeta(classifier)
# ...
```

### MotherNet Usage

MotherNet uses the same preprocessing as the TabPFN work it builds upon, but we found that using one-hot-encoding during inference improves accuracy.
Scaling of features is handled internally.

### Model Training
Full model training code is provided. Training ``MotherNet`` is possible with ``python fit_model.py mothernet``. A GPU ID can be specified with ``-g GPU_ID``. See the ``python fit_model.py mothernet -h`` and ``python fit_model.py -h`` for more options.
The results in the paper correspond to ``python fit_model.py mothernet -L 2``, though default values might change and no longer reflect the values in the paper.
Data-parallel Multi-GPU training is in principal supported using ``torchrun``.
By default, experiments are tracked using MLFlow if the ``MLFLOW_HOSTNAME`` environment variable is set. 

## Papers
This work is described in [MotherNet: A Foundational Hypernetwork for Tabular Classification](https://arxiv.org/pdf/2312.08598).
Please cite that work when using this code. As this work rests on the TabPFN work, I would suggest you also cite their [paper](https://arxiv.org/abs/2207.01848),
which also provides more background on the methodology.

## License
Copyright 2022 Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter

Additions by Andreas Mueller, 2024

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
