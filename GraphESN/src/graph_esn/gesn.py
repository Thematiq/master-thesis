import torch

from torch import nn, Tensor
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import get_laplacian, to_dense_adj
from typing import Tuple, Union, List
from sklearn.ensemble import RandomForestClassifier

from auto_esn.esn.reservoir.initialization import WeightInitializer
from auto_esn.esn.reservoir.activation import Activation, self_normalizing_default
from GraphESN.src.graph_esn.reservoir.graph import GroupOfGESNCell, DeepGESNCell
from GraphESN.src.graph_esn.readout.aggregator import LatentFeatureAggregator, mean_vertex_features


class GESNBase(nn.Module):
    def __init__(self, reservoir: nn.Module, readout,
                 aggregator: LatentFeatureAggregator = mean_vertex_features):
        super(GESNBase, self).__init__()
        self.reservoir = reservoir
        self.readout = readout
        self.aggregator = aggregator

    @staticmethod
    def __prepare_data(data: Data) -> Tuple[Tensor, Tensor]:
        x, e = data.x, data.edge_index
        e_i, e_a = get_laplacian(e, num_nodes=x.size(0), normalization='sym')
        l = to_dense_adj(edge_index=e_i, edge_attr=e_a).squeeze(dim=0)
        return x, l

    def __reservoir_forward(self, data: List[Data]) -> Tensor:
        self.reservoir.reset_hidden()
        mapped_features = []
        for graph in data:
            self.reservoir.reset_hidden()
            res = self.reservoir(*self.__prepare_data(graph))
            mapped_features.append(self.aggregator(res))
        return torch.stack(mapped_features, dim=0).cpu()

    def fit(self, data: List[Data], target: Tensor):
        mapped_features = self.__reservoir_forward(data)
        self.readout.fit(mapped_features, target)

    def forward(self, data: Union[Data, Dataset]) -> Tensor:
        if isinstance(data, Data):
            return self.readout.predict(self.__reservoir_forward([data]))
        else:
            return self.readout.predict(self.__reservoir_forward(data))

    def reset_hidden(self):
        self.reservoir.reset_hidden()

    def to_cuda(self):
        self.reservoir.to_cuda()
        # self.readout.to_cuda()


class DeepESN(GESNBase):
    def __init__(self, input_size: int = 1, hidden_size: int = 500,
                 initializer: WeightInitializer = WeightInitializer(), num_layers: int = 2,
                 activation=self_normalizing_default(),
                 conv_th: float = 1e-5, max_iter: int = 50, readout=RandomForestClassifier(n_jobs=-1, n_estimators=500),
                 aggregator: LatentFeatureAggregator = mean_vertex_features):
        super().__init__(
            reservoir=DeepGESNCell(
                input_size=input_size, hidden_size=hidden_size,
                initializer=initializer, num_layers=num_layers,
                activation=activation, conv_th=conv_th, max_iter=max_iter),
            readout=readout,
            aggregator=aggregator)


class GroupOfGESN(GESNBase):
    def __init__(self, input_size: int = 1, hidden_size: int = 250,
                 initializer: WeightInitializer = WeightInitializer(), groups=4,
                 activation: Activation = self_normalizing_default(), conv_th: float = 1e-5, max_iter: int = 50,
                 readout=RandomForestClassifier(n_jobs=-1, n_estimators=500),
                 aggregator: LatentFeatureAggregator = mean_vertex_features):
        super().__init__(
            reservoir=GroupOfGESNCell(
                input_size=input_size, hidden_size=hidden_size,
                initializer=initializer, groups=groups,
                activation=activation, conv_th=conv_th, max_iter=max_iter),
            readout=readout,
            aggregator=aggregator)


class FlexDeepGESN(GESNBase):
    def __init__(self, input_size: int = 1, hidden_size: int = 500,
                 initializer: WeightInitializer = WeightInitializer(), num_layers=2,
                 activation: Activation = self_normalizing_default(), conv_th: float = 1e-5,
                 readout=RandomForestClassifier(n_jobs=-1, n_estimators=500),
                 max_iter: int = 50, aggregator: LatentFeatureAggregator = mean_vertex_features):
        super().__init__(
            reservoir=DeepGESNCell(
                input_size=input_size, hidden_size=hidden_size,
                initializer=initializer, num_layers=num_layers,
                activation=activation, conv_th=conv_th, max_iter=max_iter),
            readout=readout,
            aggregator=aggregator)


class GroupedDeepGESN(GESNBase):
    def __init__(self, input_size: int = 1, hidden_size: int = 250,
                 initializer: WeightInitializer = WeightInitializer(), groups=2, num_layers=2,
                 activation: Activation = self_normalizing_default(), conv_th: float = 1e-5, max_iter: int = 50,
                 readout=RandomForestClassifier(n_jobs=-1, n_estimators=500), network_size=None,
                 aggregator: LatentFeatureAggregator = mean_vertex_features):
        hidden_size = hidden_size if network_size is None else network_size // sum(num_layers)
        super().__init__(
            reservoir=GroupOfGESNCell(
                input_size=input_size, hidden_size=hidden_size, groups=[
                        DeepGESNCell(input_size=input_size,
                                     hidden_size=hidden_size,
                                     initializer=initializer,
                                     num_layers=num_layers,
                                     activation=activation,
                                     conv_th=conv_th,
                                     max_iter=max_iter)
                        for _ in range(groups)],
                activation=activation, initializer=initializer,
                conv_th=conv_th, max_iter=max_iter),
            readout=readout,
            aggregator=aggregator)
