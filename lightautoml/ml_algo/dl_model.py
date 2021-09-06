"""Neural net for tabular datasets."""

import gc
import os
import uuid
from copy import copy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from optuna import Trial
from torch.optim import lr_scheduler
from transformers import AutoTokenizer
from .nn_models import DenseBaseModel, DenseModel, ResNetModel
from .tuning.optuna import OptunaTunableMixin

from ..ml_algo.base import TabularMLAlgo, TabularDataset
from ..pipelines.features.text_pipeline import _model_name_by_lang
from ..pipelines.utils import get_columns_by_role
from ..text.nn_model import TorchUniversalModel, ContEmbedder, CatEmbedder, TextBert, UniversalDataset
from ..text.trainer import Trainer
from ..text.utils import seed_everything, parse_devices, collate_dict, is_shuffle, inv_softmax, inv_sigmoid
from ..utils.logging import get_logger

from ..ml_algo.torch_based.act_funcs import TS

logger = get_logger(__name__)

model_by_name = {'dense_light': DenseBaseModel, 'dense': DenseModel, 'resnet': ResNetModel}


class TorchModel(OptunaTunableMixin, TabularMLAlgo):
    """Neural net for tabular datasets.

    default_params:

        - bs: Batch size.
        - num_workers: Number of threads for multiprocessing.
        - max_length: Max sequence length.
        - opt_params: Dict with optim params.
        - scheduler_params: Dict with scheduler params.
        - is_snap: Use snapshots.
        - snap_params: Dict with SE parameters.
        - init_bias: Init last linear bias by mean target values.
        - n_epochs: Number of training epochs.
        - input_bn: Use 1d batch norm for input data.
        - emb_dropout: Dropout probability.
        - emb_ratio: Ratio for embedding size = (x + 1) // emb_ratio.
        - max_emb_size: Max embedding size.
        - bert_name: Name of HuggingFace transformer model.
        - pooling: Type of pooling strategy for bert model.
        - device: Torch device or str.
        - use_cont: Use numeric data.
        - use_cat: Use category data.
        - use_text: Use text data.
        - lang: Text language.
        - deterministic: CUDNN backend.
        - multigpu: Use Data Parallel.
        - path_to_save: Path to save model checkpoints,
          ``None`` - stay in memory.
        - random_state: Random state to take subsample.
        - verbose_inside: Number of steps between
          verbose inside epoch or ``None``.
        - verbose: Verbose every N epochs.

    freeze_defaults:

        - ``True`` :  params may be rewrited depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.

    timer: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.

    """
    _name: str = 'TorchNN'

    _default_params = {
        'bs': 16,
        'num_workers': 4,
        'max_length': 256,
        'opt': torch.optim.Adam,
        'opt_params': {'lr': 1e-4},
        'scheduler_params': {'patience': 5, 'factor': 0.5, 'verbose': True},
        'is_snap': False,
        'snap_params': {'k': 1, 'early_stopping': True, 'patience': 1, 'swa': False},
        'init_bias': True,
        'n_epochs': 20,
        'input_bn': False,
        'emb_dropout': 0.1,
        'emb_ratio': 3,
        'max_emb_size': 256,
        'bert_name': None,
        'pooling': 'cls',
        'device': torch.device('cuda:0'),
        'use_cont': True,
        'use_cat': True,
        'use_text': True,
        'lang': 'en',
        'deterministic': True,
        'multigpu': False,
        'random_state': 42,
        'efficient': False,
        'model': 'dense_light',
        'path_to_save': os.path.join('./models/', 'model'),
        'verbose_inside': None,
        'verbose': 1,
    }

    def _infer_params(self):
        # assert False
        if self.params['path_to_save'] is not None:
            self.path_to_save = os.path.relpath(self.params['path_to_save'])
            if not os.path.exists(self.path_to_save):
                os.makedirs(self.path_to_save)
        else:
            self.path_to_save = None

        params = copy(self.params)
        if params['bert_name'] is None:
            params['bert_name'] = _model_name_by_lang[params['lang']]

        params['loss'] = self.task.losses['torch'].loss
        params['metric'] = self.task.losses['torch'].metric_func

        is_text = (len(params["text_features"]) > 0) and (params["use_text"]) and (params['device'].type == 'cuda')
        is_cat = (len(params["cat_features"]) > 0) and (params["use_cat"])
        is_cont = (len(params["cont_features"]) > 0) and (params["use_cont"])

        model = Trainer(
            net=TorchUniversalModel,
            net_params={**params, **{
                'loss': params['loss'],
                'task': self.task,
                'n_out': params['n_out'],
                'cont_embedder': ContEmbedder if is_cont else None,
                'cont_params': {'num_dims': params['cont_dim'],
                                'input_bn': params['input_bn']} if is_cont else None,
                'cat_embedder': CatEmbedder if is_cat else None,
                'cat_params': {'cat_dims': params['cat_dims'], 'emb_dropout': params['emb_dropout'],
                               'emb_ratio': params['emb_ratio'],
                               'max_emb_size': params['max_emb_size']} if is_cat else None,
                'text_embedder': TextBert if is_text else None,
                'text_params': {'model_name': params['bert_name'],
                                'pooling': params['pooling']} if is_text else None,
                'bias': params['bias'],
                'torch_model': model_by_name[params["model"]],
            }},
            opt=params['opt'],
            opt_params=params['opt_params'],
            n_epochs=params['n_epochs'],
            device=params['device'],
            device_ids=params['device_ids'],
            is_snap=params['is_snap'],
            snap_params=params['snap_params'],
            sch=lr_scheduler.ReduceLROnPlateau,
            scheduler_params=params['scheduler_params'],
            verbose=params['verbose'],
            verbose_inside=params['verbose_inside'],
            metric=params['metric'],
            apex=False,
        )

        self.train_params = {
            'dataset': UniversalDataset, 'bs': params['bs'], 'num_workers': params['num_workers'],
            'tokenizer': AutoTokenizer.from_pretrained(params['bert_name'], use_fast=False) if is_text else None,
            'max_length': params['max_length']
        }

        return model

    @staticmethod
    def get_mean_target(target, task_name):
        bias = np.array(target.mean(axis=0)).reshape(1, -1).astype(float) if (task_name != 'multiclass') else \
            np.unique(target, return_counts=True)[1]
        bias = inv_sigmoid(bias) if (task_name == 'binary') or (task_name == 'multilabel') else inv_softmax(bias) if (
                task_name == 'multiclass') else bias

        bias[bias == np.inf] = np.nanmax(bias[bias != np.inf])
        bias[bias == -np.inf] = np.nanmin(bias[bias != -np.inf])
        bias[bias == np.NaN] = np.nanmean(bias[bias != np.NaN])

        return bias

    def init_params_on_input(self, train_valid_iterator) -> dict:
        """Get model parameters depending on dataset parameters.

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Parameters of model.

        """

        suggested_params = copy(self.default_params)
        suggested_params['device'], suggested_params['device_ids'] = parse_devices(suggested_params['device'],
                                                                                   suggested_params['multigpu'])

        task_name = train_valid_iterator.train.task.name
        target = train_valid_iterator.train.target
        suggested_params['n_out'] = 1 if task_name != 'multiclass' else np.max(target) + 1

        cat_dims = []
        suggested_params['cat_features'] = get_columns_by_role(train_valid_iterator.train, 'Category')
        for cat_feature in suggested_params['cat_features']:
            num_unique_categories = max(train_valid_iterator.train[:, cat_feature].data)
            cat_dims.append(num_unique_categories)
        suggested_params['cat_dims'] = cat_dims

        suggested_params['cont_features'] = get_columns_by_role(train_valid_iterator.train, 'Numeric')
        suggested_params['cont_dim'] = len(suggested_params['cont_features'])

        suggested_params['text_features'] = get_columns_by_role(train_valid_iterator.train, 'Text')
        suggested_params['bias'] = self.get_mean_target(target, task_name) if suggested_params['init_bias'] else None

        return suggested_params

    def get_dataloaders_from_dicts(self, data_dict):
        logger.debug(f'n text: {len(self.params["text_features"])} ')
        logger.debug(f'n cat: {len(self.params["cat_features"])} ')
        logger.debug(f'n cont: {self.params["cont_dim"]} ')

        datasets = {}
        for stage, value in data_dict.items():
            data = {
                name: value.data[cols].values for name, cols in
                zip(
                    ['text', 'cat', 'cont'],
                    [
                        self.params["text_features"],
                        self.params["cat_features"],
                        self.params["cont_features"]
                    ]
                ) if len(cols) > 0
            }

            datasets[stage] = self.train_params['dataset'](
                data=data,
                y=value.target.values if stage != 'test' else np.ones(len(value.data)),
                w=value.weights.values if value.weights is not None else np.ones(
                    len(value.data)),
                tokenizer=self.train_params['tokenizer'],
                max_length=self.train_params['max_length'],
                stage=stage
            )

        dataloaders = {stage: torch.utils.data.DataLoader(datasets[stage],
                                                          batch_size=self.train_params['bs'],
                                                          shuffle=is_shuffle(stage),
                                                          num_workers=self.train_params['num_workers'],
                                                          collate_fn=collate_dict,
                                                          pin_memory=False) for stage, value in data_dict.items()}
        return dataloaders

    def fit_predict_single_fold(self, train, valid):
        """Implements training and prediction on single fold.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Tuple (model, predicted_values).

        """
        seed_everything(self.params['random_state'], self.params['deterministic'])
        task_name = train.task.name
        target = train.target
        self.params['bias'] = self.get_mean_target(target, task_name) if self.params['init_bias'] else None
        model = self._infer_params()

        model_path = os.path.join(self.path_to_save,
                                  f'{uuid.uuid4()}.pickle') if self.path_to_save is not None else None
        # init datasets
        dataloaders = self.get_dataloaders_from_dicts({'train': train.to_pandas(), 'val': valid.to_pandas()})

        val_pred = model.fit(dataloaders)

        if model_path is None:
            model_path = model.state_dict(model_path)
        else:
            model.state_dict(model_path)

        model.clean()
        del dataloaders, model
        gc.collect()
        torch.cuda.empty_cache()
        return model_path, val_pred

    def predict_single_fold(self, model: any, dataset: TabularDataset) -> np.ndarray:
        """Predict target values for dataset.

        Args:
            model: Neural net object or dict or str.
            dataset: Test dataset.

        Return:
            Predicted target values.

        """

        seed_everything(self.params['random_state'], self.params['deterministic'])
        dataloaders = self.get_dataloaders_from_dicts({'test': dataset.to_pandas()})

        if isinstance(model, (str, dict)):
            model = self._infer_params().load_state(model)

        pred = model.predict(dataloaders['test'], 'test')

        model.clean()
        del dataloaders, model
        gc.collect()
        torch.cuda.empty_cache()

        return pred

    def sample_params_values(self, trial: Trial, suggested_params: Dict, estimated_n_trials: int) -> Dict:
        """Sample hyperparameters from suggested.

        Args:
            trial: Optuna trial object.
            suggested_params: Dict with parameters.
            estimated_n_trials: Maximum number of hyperparameter estimations.

        Returns:
            dict with sampled hyperparameters.

        """
        logger.debug('Suggested parameters:')
        logger.debug(suggested_params)

        trial_values = copy(suggested_params)

        trial_values['model'] = trial.suggest_categorical(
            'model',
            ['dense_light', 'dense', 'resnet']
        )

        if 'is_cat' in self.params and self.params['is_cat'] and\
                len(self.params['cat_dims']) > 0:
            trial_values['emb_dropout'] = trial.suggest_uniform(
                'emb_dropout',
                low=0,
                high=0.2)
            trial_values['emb_ratio'] = trial.suggest_int(
                'emb_ratio',
                low=2,
                high=6)

        opt_params = {'lr': trial.suggest_loguniform(
            'lr',
            low=1e-5,
            high=1e-2
        ), 'weight_decay': trial.suggest_loguniform(
            'weight_decay',
            low=1e-4,
            high=1e-2
        )}
        trial_values['opt_params'] = opt_params

        trial_values['opt'] = trial.suggest_categorical(
            'opt',
            [torch.optim.Adam, torch.optim.AdamW]
        )

        trial_values['act_fun'] = trial.suggest_categorical(
            'act_fun',
            [nn.ReLU, TS]
        )

        trial_values['init_bias'] = trial.suggest_categorical(
            'init_bias',
            [True, False]
        )

        if trial_values['model'] == 'dense_light':
            trial_values['num_layers'] = trial.suggest_int(
                'num_layers',
                low=1,
                high=8
            )

            hidden_size = ()
            drop_rate = ()
            hid_high = 1024

            if trial_values['num_layers'] > 4:
                hid_high = 512

            for layer in range(trial_values['num_layers']):
                hidden_name = 'hidden_size_' + str(layer)
                drop_name = 'drop_rate_' + str(layer)

                trial_values[hidden_name] = trial.suggest_int(
                    hidden_name,
                    low=1,
                    high=hid_high
                )
                trial_values[drop_name] = trial.suggest_uniform(
                    drop_name,
                    low=0.0,
                    high=0.5
                )

                hidden_size = hidden_size + (trial_values[hidden_name],)
                drop_rate = drop_rate + (trial_values[drop_name],)

            trial_values['hidden_size'] = hidden_size
            trial_values['drop_rate'] = drop_rate

            trial_values['noise_std'] = trial.suggest_loguniform(
                'noise_std',
                low=1e-5,
                high=1e-2
            )

        elif trial_values['model'] == 'dense':
            trial_values['num_blocks'] = trial.suggest_int(
                'num_blocks',
                low=1,
                high=8
            )

            block_config = ()
            drop_rate = ()

            block_high = 8

            if trial_values['num_blocks'] > 4:
                block_high = 4

            for block in range(trial_values['num_blocks']):
                block_name = 'block_size_' + str(block)
                drop_name = 'drop_rate_' + str(block)

                trial_values[block_name] = trial.suggest_int(
                    block_name,
                    low=1,
                    high=block_high
                )
                trial_values[drop_name] = trial.suggest_uniform(
                    drop_name,
                    low=0.0,
                    high=0.5
                )

                block_config = block_config + (trial_values[block_name],)
                drop_rate = drop_rate + (trial_values[drop_name],)

            trial_values['block_config'] = block_config
            trial_values['drop_rate'] = drop_rate

            trial_values['num_init_features'] = trial.suggest_int(
                'num_init_features',
                low=1,
                high=1024
            )

            trial_values['compression'] = trial.suggest_uniform(
                'compression',
                low=0.0,
                high=0.9
            )

            gr_high = 64
            bn_size = 32
            if trial_values['num_blocks'] > 4:
                gr_high = 32
                bn_size = 16

            trial_values['growth_rate'] = trial.suggest_int(
                'growth_rate',
                low=8,
                high=gr_high
            )

            trial_values['bn_size'] = trial.suggest_int(
                'bn_size',
                low=2,
                high=bn_size
            )

        elif trial_values['model'] == 'resnet':
            trial_values['num_layers'] = trial.suggest_int(
                'num_layers',
                low=1,
                high=16
            )

            hidden_factor = ()
            drop_rate = ()
            hid_high = 40

            if trial_values['num_layers'] > 5:
                hid_high = 20

            for layer in range(trial_values['num_layers']):
                hidden_name = 'hidden_factor_' + str(layer)
                drop_name = 'drop_rate_' + str(layer)

                trial_values[hidden_name] = trial.suggest_uniform(
                    hidden_name,
                    low=1.0,
                    high=hid_high
                )
                trial_values[drop_name + '_1'] = trial.suggest_uniform(
                    drop_name,
                    low=0.0,
                    high=0.5
                )
                trial_values[drop_name + '_2'] = trial.suggest_uniform(
                    drop_name,
                    low=0.0,
                    high=0.5
                )

                hidden_factor = hidden_factor + (trial_values[hidden_name],)
                drop_rate = drop_rate + ((trial_values[drop_name + '_1'], trial_values[drop_name + '_2']),)

            trial_values['hidden_factor'] = hidden_factor
            trial_values['drop_rate'] = drop_rate

            trial_values['noise_std'] = trial.suggest_loguniform(
                'noise_std',
                low=1e-5,
                high=1e-2
            )

        return trial_values
