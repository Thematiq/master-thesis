import torch

from time import time
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from graph_esn.gesn import GroupedDeepGESN
from auto_esn.esn.reservoir.activation import self_normalizing_default
from graph_esn.readout.aggregator import sum_vertex_features
from auto_esn.esn.reservoir.initialization import WeightInitializer, default_hidden


dataset = TUDataset(root='data', name='Mutagenicity')

train_data, test_data, y_train, y_test = train_test_split(dataset, dataset.y, test_size=0.2, stratify=dataset.y,
                                                          random_state=42)

model = GroupedDeepGESN(input_size=dataset.num_features, hidden_size=50, activation=self_normalizing_default(),
                        num_layers=3, groups=3, aggregator=sum_vertex_features,
                        initializer=WeightInitializer(
                            weight_hh_init=default_hidden(spectral_radius=0.5)
                        ), conv_th=1e-3)

# print('USING CUDA')
# model.to_cuda()
# train_data = [x.to('cuda:0') for x in train_data]
# test_data = [x.to('cuda:0') for x in test_data]

y_true = y_test.numpy()
start = time()
with torch.no_grad():
    model.fit(train_data, y_train)
    y_pred = model(test_data)
end = time()
print(classification_report(y_pred, y_true))
print(f'Time {end - start:.2f}s')
