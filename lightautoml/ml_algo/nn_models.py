from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import math

from lightautoml.ml_algo.torch_based.act_funcs import TS


class GaussianNoise(nn.Module):
    def __init__(self, stddev, device):
        super().__init__()
        self.stddev = stddev
        self.device = device

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable(torch.randn(din.size()).to(self.device) * self.stddev)
        return din


class UniformNoise(nn.Module):
    def __init__(self, stddev, device):
        super().__init__()
        self.stddev = stddev
        self.device = device

    def forward(self, din):
        if self.training:
            return din + torch.autograd.Variable((torch.rand(din.size()).to(self.device) - 0.5) * self.stddev)
        return din


class DenseBaseModel(nn.Module):
    def __init__(self, n_in, n_out=1, hidden_size=(512, 750,), drop_rate=(0.1, 0.1,),
                 bias=None, noise_std=0.05, act_fun=nn.ReLU, **kwargs):
        super(DenseBaseModel, self).__init__()
        assert len(hidden_size) == len(drop_rate), 'Wrong number hidden_sizes/drop_rates. Must be equal.'

        self.features = nn.Sequential(OrderedDict([]))
        num_features = n_in
        for i, hid_size in enumerate(hidden_size):
            block = DenseBaseModel._DenseBlock(
                n_in=num_features,
                n_out=hid_size,
                drop_rate=drop_rate[i],
                noise_std=noise_std,
                act_fun=act_fun
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = n_in + hid_size

        self.fc = nn.Linear(hidden_size[-1], n_out)

        if bias is not None:
            print('init bias!')
            bias = torch.Tensor(bias)
            self.fc.bias.data = bias
            self.fc.weight.data = torch.Variable(torch.zeros(n_out, int(hidden_size[-1]) + n_in))

    def forward(self, x):
        input = x.detach().clone()
        for name, layer in self.features.named_children():
            if name != 'denseblock1':
                x = torch.cat([x, input], 1)
            x = layer(x)

        logits = self.fc(x)
        return logits.view(logits.shape[0], -1)

    class _DenseBlock(nn.Module):
        def __init__(self, n_in, n_out, drop_rate, noise_std, act_fun):
            super(DenseBaseModel._DenseBlock, self).__init__()
            self.norm = nn.BatchNorm1d(n_in)
            self.drop = nn.Dropout(p=drop_rate)
            self.noise = GaussianNoise(noise_std, torch.device('cuda:0'))
            self.dense = nn.Linear(n_in, n_out)
            self.act = act_fun()

        def forward(self, x):
            x = self.noise(self.drop(self.norm(x)))
            x = self.act(self.dense(x))
            return x


class DenseModel(nn.Module):
    def __init__(self, n_in, n_out=1, block_config=(2, 2), drop_rate=(0.1, 0.1), num_init_features=512,
                 compression=0.5, growth_rate=16, bn_size=4, bias=None, act_fun=nn.ReLU,
                 efficient=False, **kwargs):

        super(DenseModel, self).__init__()
        assert 0 < compression <= 1, 'compression of densenet should be between 0 and 1'

        self.features = nn.Sequential(OrderedDict([
            ('dense0', nn.Linear(n_in, num_init_features)),
        ]))
        self.features.add_module('norm0', nn.BatchNorm1d(num_init_features))
        self.features.add_module('act0', act_fun())

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseModel._DenseBlock(
                num_layers=num_layers,
                n_in=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate[i],
                act_fun=act_fun,
                efficient=efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = self._Transition(n_in=num_features,
                                         n_out=int(num_features * compression),
                                         act_fun=act_fun)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        self.features.add_module('norm_final', nn.BatchNorm1d(num_features))

        self.fc = nn.Linear(num_features, n_out)
        if bias is not None:
            print('init bias!')
            bias = torch.Tensor(bias)
            self.fc.bias.data = bias
            self.fc.weight.data = torch.Variable(torch.zeros(n_out, num_features + n_in))

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = out.view(out.shape[0], -1)
        return out

    def _bn_function_factory(norm, act, dense):
        def bn_function(*inputs):
            concated_features = torch.cat(inputs, 1)
            bottleneck_output = dense(act(norm(concated_features)))
            return bottleneck_output

        return bn_function

    class _DenseLayer(nn.Module):
        def __init__(self, n_in, growth_rate, bn_size, drop_rate, act_fun, efficient=False):
            super(DenseModel._DenseLayer, self).__init__()
            self.add_module('norm1', nn.BatchNorm1d(n_in)),
            self.add_module('act1', act_fun()),
            self.add_module('dense1', nn.Linear(n_in, bn_size * growth_rate)),
            self.add_module('norm2', nn.BatchNorm1d(bn_size * growth_rate)),
            self.add_module('act2', act_fun()),
            self.add_module('dense2', nn.Linear(bn_size * growth_rate, growth_rate)),
            self.drop_rate = drop_rate
            self.efficient = efficient

        def forward(self, *prev_features):
            bn_function = DenseModel._bn_function_factory(self.norm1, self.act1, self.dense1)
            if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
                bottleneck_output = cp.checkpoint(bn_function, *prev_features)
            else:
                bottleneck_output = bn_function(*prev_features)
            new_features = self.dense2(self.act2(self.norm2(bottleneck_output)))

            if self.drop_rate > 0:
                new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
            return new_features

    class _Transition(nn.Sequential):
        def __init__(self, n_in, n_out, act_fun):
            super(DenseModel._Transition, self).__init__()
            self.add_module('norm', nn.BatchNorm1d(n_in))
            self.add_module('act', act_fun())
            self.add_module('conv', nn.Linear(n_in, n_out))

    class _DenseBlock(nn.Module):
        def __init__(self, num_layers, n_in, bn_size, growth_rate, drop_rate, act_fun, efficient=False):
            super(DenseModel._DenseBlock, self).__init__()
            for i in range(num_layers):
                layer = DenseModel._DenseLayer(
                    n_in + i * growth_rate,
                    growth_rate=growth_rate,
                    bn_size=bn_size,
                    drop_rate=drop_rate,
                    act_fun=act_fun,
                    efficient=efficient,
                )
                self.add_module('denselayer%d' % (i + 1), layer)

        def forward(self, init_features):
            features = [init_features]
            for name, layer in self.named_children():
                new_features = layer(*features)
                features.append(new_features)
            return torch.cat(features, 1)


class ResNetModel(nn.Module):
    def __init__(self, n_in, n_out=1, hidden_factor=(5, 5, 5), drop_rate=((0.1, 0.1), (0.1, 0.1), (0.1, 0.1)),
                 bias=None, noise_std=0.05, act_fun=nn.ReLU, **kwargs):
        super(ResNetModel, self).__init__()

        self.features = nn.Sequential(OrderedDict([]))
        for i, hid_factor in enumerate(hidden_factor):
            block = ResNetModel._ResNetBlock(
                n_in=n_in,
                hidden_factor=hid_factor,
                n_out=n_in,
                drop_rate=drop_rate[i],
                noise_std=noise_std,
                act_fun=act_fun
            )
            self.features.add_module('resnetblock%d' % (i + 1), block)

        self.fc = nn.Linear(n_in, n_out)

        if bias is not None:
            print('init bias!')
            bias = torch.Tensor(bias)
            self.fc.bias.data = bias
            self.fc.weight.data = torch.Variable(torch.zeros(n_out, int(hidden_size[-1]) + n_in))

    class _ResNetBlock(nn.Module):
        def __init__(self, n_in, hidden_factor, n_out, drop_rate, noise_std, act_fun):
            super(ResNetModel._ResNetBlock, self).__init__()
            self.norm = nn.BatchNorm1d(n_in)
            self.act1 = act_fun()
            self.drop1 = nn.Dropout(p=drop_rate[0])
            self.noise = GaussianNoise(noise_std, torch.device('cuda:0'))
            self.dense1 = nn.Linear(n_in, int(n_in * hidden_factor))
            self.act2 = act_fun()
            self.drop2 = nn.Dropout(p=drop_rate[1])
            self.dense2 = nn.Linear(int(n_in * hidden_factor), n_out)

        def forward(self, x):
            x = self.noise(self.drop1(self.act1(self.norm(x))))
            x = self.dense2(self.drop2(self.act2(self.dense1(x))))
            return x

    def predict(self, x):
        identity = x
        for name, layer in self.features.named_children():
            if name != 'resnetblock1':
                x += identity
                identity = x
            x = layer(x)

        logits = self.fc(x)
        return logits.view(logits.shape[0], -1)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = out.view(out.shape[0], -1)
        return out
