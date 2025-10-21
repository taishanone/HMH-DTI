import torch
from torch.nn import Linear, ReLU, Dropout,Sequential,Conv1d,Embedding,MaxPool1d
from torch_geometric.nn import GCNConv, DenseGCNConv, GATConv,DenseGATConv,global_mean_pool as gep
from torch_geometric.utils import dropout_adj
import math

vector_operations = {
    "cat": (lambda x, y: torch.cat((x, y), -1), lambda dim: 2 * dim),
    "add": (torch.add, lambda dim: dim),
    "sub": (torch.sub, lambda dim: dim),
    "mul": (torch.mul, lambda dim: dim),
    "combination1": (lambda x, y: torch.cat((x, y, torch.add(x, y)), -1), lambda dim: 3 * dim),
    "combination2": (lambda x, y: torch.cat((x, y, torch.sub(x, y)), -1), lambda dim: 3 * dim),
    "combination3": (lambda x, y: torch.cat((x, y, torch.mul(x, y)), -1), lambda dim: 3 * dim),
    "combination4": (lambda x, y: torch.cat((torch.add(x, y), torch.sub(x, y)), -1), lambda dim: 2 * dim),
    "combination5": (lambda x, y: torch.cat((torch.add(x, y), torch.mul(x, y)), -1), lambda dim: 2 * dim),
    "combination6": (lambda x, y: torch.cat((torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 2 * dim),
    "combination7": (lambda x, y: torch.cat((torch.add(x, y), torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 3 * dim),
    "combination8": (lambda x, y: torch.cat((x, y, torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 4 * dim),
    "combination9": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.mul(x, y)), -1), lambda dim: 4 * dim),
    "combination10": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.sub(x, y)), -1), lambda dim: 4 * dim),
    "combination11": (lambda x, y: torch.cat((x, y, torch.add(x, y), torch.sub(x, y), torch.mul(x, y)), -1), lambda dim: 5 * dim),
    "combination12": (lambda x, y: torch.cat((torch.add(x, y), torch.sub(x, y)), -1), lambda dim: 4 * dim)
}
class CNNBlock(torch.nn.Module):
    def __init__(self,embedding_vocab,channal_input_dim,channal_output_dim):
        super(CNNBlock, self).__init__()
        self.embedding_smiles = Embedding(embedding_vocab,channal_input_dim)
        self.CNN_smiles =Sequential(
            Conv1d(in_channels=channal_input_dim, out_channels=512, kernel_size=4,padding=1),
            ReLU(),
            Conv1d(in_channels=512, out_channels=channal_output_dim, kernel_size=4,padding=1),
            ReLU()
        )
        self.Drug_max_pool=MaxPool1d(100-3-3+3)

    def forward(self, smiles_embedding):
        smiles_embedding=self.embedding_smiles(smiles_embedding).permute(0,2,1)
        smiles_embedding=self.CNN_smiles(smiles_embedding)
        drug_smiles_transform_embedding=self.Drug_max_pool(smiles_embedding).squeeze(2)
        return drug_smiles_transform_embedding

class CNNBlock_target(torch.nn.Module):
    def __init__(self,embedding_vocab,channal_input_dim,channal_output_dim):
        super(CNNBlock_target, self).__init__()
        self.embedding_target = Embedding(embedding_vocab,channal_input_dim)
        self.Protein_CNNs = Sequential(
                    Conv1d(in_channels=channal_input_dim, out_channels=40, kernel_size=4),
                    ReLU(),
                    Conv1d(in_channels=40, out_channels=80, kernel_size=8),
                    ReLU(),
                    Conv1d(in_channels=80, out_channels=channal_output_dim, kernel_size=12),
                    ReLU(),
                )
        self.Protein_max_pool = MaxPool1d(1000-4-8-12+3)

    def forward(self, target_seq):
        target_seq=self.embedding_target(target_seq).permute(0,2,1)
        target_seq=self.Protein_CNNs(target_seq)
        target_seq_transform_embedding=self.Protein_max_pool(target_seq).squeeze(2)
        return target_seq_transform_embedding

class LinearBlock(torch.nn.Module):
    def __init__(self, linear_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[]):
        super(LinearBlock, self).__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(len(linear_layers_dim) - 1):
            layer = Linear(linear_layers_dim[i], linear_layers_dim[i + 1])
            self.layers.append(layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x):
        output = x
        embeddings = []
        for layer_index in range(len(self.layers)):
            output = self.layers[layer_index](output)
            if layer_index in self.relu_layers_index:
                output = self.relu(output)
            if layer_index in self.dropout_layers_index:
                output = self.dropout(output)
        embeddings.append(output)
        return embeddings


class DenseGATBlock(torch.nn.Module):
    def __init__(self, GAT_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[],
                 supplement_mode=None, heads=1):
        super(DenseGATBlock, self).__init__()

        self.GAT_layers = torch.nn.ModuleList()
        for i in range(len(GAT_layers_dim) - 1):
            if supplement_mode is not None and i == 1:
                self.supplement_func, supplement_dim_func = vector_operations[supplement_mode]
                GAT_layer_input = supplement_dim_func(GAT_layers_dim[i])
            else:
                GAT_layer_input = GAT_layers_dim[i]
            GAT_layer = DenseGATConv(
                in_channels=GAT_layer_input, out_channels=GAT_layers_dim[i + 1] // heads,
                heads=heads, concat=True, dropout=dropout_rate
            )
            self.GAT_layers.append(GAT_layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index


    def forward(self, x, adj, supplement_x=None):
        output = x
        embeddings = []
        for GAT_layer_index in range(len(self.GAT_layers)):
            if supplement_x is not None and GAT_layer_index == 1:
                supplement_x = torch.unsqueeze(supplement_x, 0)
                output = self.supplement_func(output, supplement_x)


            output = self.GAT_layers[GAT_layer_index](output, adj,add_loop=True)

            if GAT_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if GAT_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
        embeddings.append(torch.squeeze(output, dim=0))
        return embeddings

class DenseGCNBlock(torch.nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[], supplement_mode=None):
        super(DenseGCNBlock, self).__init__()

        self.conv_layers = torch.nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            if supplement_mode is not None and i == 1:
                self.supplement_func, supplement_dim_func = vector_operations[supplement_mode]
                conv_layer_input = supplement_dim_func(gcn_layers_dim[i])
            else:
                conv_layer_input = gcn_layers_dim[i]
            conv_layer = DenseGCNConv(conv_layer_input, gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)
        
        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, adj, supplement_x=None):
        output = x
        embeddings = [x]
        for conv_layer_index in range(len(self.conv_layers)):
            if supplement_x is not None and conv_layer_index == 1:
                supplement_x = torch.unsqueeze(supplement_x, 0)
                output = self.supplement_func(output, supplement_x)
            output = self.conv_layers[conv_layer_index](output, adj, add_loop=False)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(torch.squeeze(output, dim=0))
        return embeddings


class GATBlock(torch.nn.Module):
    def __init__(self, gcn_layers_dim,dropout_rate=0, relu_layers_index=[], dropout_layers_index=[],
                 supplement_mode=None, heads=1):
        super(GATBlock, self).__init__()

        self.GAT_layers = torch.nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            if supplement_mode is not None and i == 1:
                self.supplement_func, supplement_dim_func = vector_operations[supplement_mode]
                GAT_layer_input = supplement_dim_func(gcn_layers_dim[i])
            if supplement_mode is not None and i == 2:
                self.supplement_func, supplement_dim_func = vector_operations[supplement_mode]
                GAT_layer_input = supplement_dim_func(gcn_layers_dim[i])
            if i!=1 and i!=2:
                GAT_layer_input = gcn_layers_dim[i]


            GAT_layer = GATConv(
                in_channels=GAT_layer_input,
                out_channels=gcn_layers_dim[i + 1] // heads,
                heads=heads,
                concat=True,
                dropout=dropout_rate
            )
            self.GAT_layers.append(GAT_layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch, supplement_x=None,supplement_y=None,count=None):
        output = x
        embeddings = []

        for GAT_layer_index in range(len(self.GAT_layers)):
            if supplement_x is not None and GAT_layer_index == 1:

                output = self.supplement_func(output, supplement_x)
            if supplement_y is not None and GAT_layer_index == 2:

                output=self.supplement_func(output,supplement_y)

            output = self.GAT_layers[GAT_layer_index](output, edge_index)

            if GAT_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if GAT_layer_index in self.dropout_layers_index:
                output = self.dropout(output)

        embeddings.append(gep(output, batch))
        return embeddings

class without_muti_model_GATBlock(torch.nn.Module):
    def __init__(self, gcn_layers_dim,dropout_rate=0, relu_layers_index=[], dropout_layers_index=[],
                 supplement_mode=None, heads=1):
        super(without_muti_model_GATBlock, self).__init__()

        self.GAT_layers = torch.nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            if supplement_mode is not None and i == 1:
                self.supplement_func, supplement_dim_func = vector_operations[supplement_mode]
                GAT_layer_input = supplement_dim_func(gcn_layers_dim[i])
            else:
                GAT_layer_input = gcn_layers_dim[i]


            GAT_layer = GATConv(
                in_channels=GAT_layer_input,
                out_channels=gcn_layers_dim[i + 1] // heads,
                heads=heads,
                concat=True,
                dropout=dropout_rate
            )
            self.GAT_layers.append(GAT_layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch, supplement_x=None,count=None):
        output = x
        embeddings = []

        for GAT_layer_index in range(len(self.GAT_layers)):
            if supplement_x is not None and GAT_layer_index == 1:
                output = self.supplement_func(output, supplement_x)

            output = self.GAT_layers[GAT_layer_index](output, edge_index)

            if GAT_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if GAT_layer_index in self.dropout_layers_index:
                output = self.dropout(output)

        embeddings.append(gep(output, batch))
        return embeddings

class without_hete_net_GATBlock(torch.nn.Module):
    def __init__(self, gcn_layers_dim,dropout_rate=0, relu_layers_index=[], dropout_layers_index=[],
                 supplement_mode=None, heads=1):
        super(without_hete_net_GATBlock, self).__init__()

        self.GAT_layers = torch.nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            if supplement_mode is not None and i == 2:
                self.supplement_func, supplement_dim_func = vector_operations[supplement_mode]
                GAT_layer_input = supplement_dim_func(gcn_layers_dim[i])
            else:
                GAT_layer_input = gcn_layers_dim[i]


            GAT_layer = GATConv(
                in_channels=GAT_layer_input,
                out_channels=gcn_layers_dim[i + 1] // heads,
                heads=heads,
                concat=True,
                dropout=dropout_rate
            )
            self.GAT_layers.append(GAT_layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch, supplement_y=None,count=None):
        output = x
        embeddings = []

        for GAT_layer_index in range(len(self.GAT_layers)):
            if supplement_y is not None and GAT_layer_index == 2:
                output = self.supplement_func(output, supplement_y)

            output = self.GAT_layers[GAT_layer_index](output, edge_index)

            if GAT_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if GAT_layer_index in self.dropout_layers_index:
                output = self.dropout(output)

        embeddings.append(gep(output, batch))
        return embeddings

class join_model_hete_GATBlock(torch.nn.Module):
    def __init__(self, gcn_layers_dim,dropout_rate=0, relu_layers_index=[], dropout_layers_index=[],
                 supplement_mode=None, heads=1):
        super(join_model_hete_GATBlock, self).__init__()

        self.GAT_layers = torch.nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            if supplement_mode is not None and i == 1:
                self.supplement_func, supplement_dim_func = vector_operations[supplement_mode]
                GAT_layer_input = supplement_dim_func(gcn_layers_dim[i])
            else:
                GAT_layer_input = gcn_layers_dim[i]


            GAT_layer = GATConv(
                in_channels=GAT_layer_input,
                out_channels=gcn_layers_dim[i + 1] // heads,
                heads=heads,
                concat=True,
                dropout=dropout_rate
            )
            self.GAT_layers.append(GAT_layer)

        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch, supplement_x=None,supplement_y=None,count=None):
        output = x
        embeddings = []

        for GAT_layer_index in range(len(self.GAT_layers)):
            if supplement_x is not None and GAT_layer_index == 1:

                output = self.supplement_func(output, supplement_x)
                output = self.supplement_func(output, supplement_y)

            output = self.GAT_layers[GAT_layer_index](output, edge_index)

            if GAT_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if GAT_layer_index in self.dropout_layers_index:
                output = self.dropout(output)

        embeddings.append(gep(output, batch))
        return embeddings

class GCNBlock(torch.nn.Module):
    def __init__(self, gcn_layers_dim, dropout_rate=0, relu_layers_index=[], dropout_layers_index=[], supplement_mode=None):
        super(GCNBlock, self).__init__()

        self.conv_layers = torch.nn.ModuleList()
        for i in range(len(gcn_layers_dim) - 1):
            if supplement_mode is not None and i == 1:
                self.supplement_func, supplement_dim_func = vector_operations[supplement_mode]
                conv_layer_input = supplement_dim_func(gcn_layers_dim[i])
            else:
                conv_layer_input = gcn_layers_dim[i]
            conv_layer = GCNConv(conv_layer_input, gcn_layers_dim[i + 1])
            self.conv_layers.append(conv_layer)
        
        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.relu_layers_index = relu_layers_index
        self.dropout_layers_index = dropout_layers_index

    def forward(self, x, edge_index, edge_weight, batch, supplement_x=None):
        output = x
        embeddings = [x]
        for conv_layer_index in range(len(self.conv_layers)):
            if supplement_x is not None and conv_layer_index == 1:
                output = self.supplement_func(output, supplement_x)
            output = self.conv_layers[conv_layer_index](output, edge_index, edge_weight)
            if conv_layer_index in self.relu_layers_index:
                output = self.relu(output)
            if conv_layer_index in self.dropout_layers_index:
                output = self.dropout(output)
            embeddings.append(gep(output, batch))
        return embeddings


class DenseGATModel(torch.nn.Module):
    def __init__(self, layers_dim, edge_dropout_rate=0, supplement_mode=None, heads=1):
        super(DenseGATModel, self).__init__()
        print('DenseGATModel Loaded')

        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_GAT = DenseGATBlock(
            layers_dim, 0.1, relu_layers_index=range(self.num_layers),
            dropout_layers_index=range(self.num_layers), supplement_mode=supplement_mode, heads=heads
        )

    def forward(self, graph, substitution_x=None, supplement_x=None):

        xs, adj=(substitution_x if substitution_x is not None else graph.x),graph.adj


        embeddings = self.graph_GAT(xs, adj, supplement_x=supplement_x)

        return embeddings

class DenseGCNModel(torch.nn.Module):
    def __init__(self, layers_dim, edge_dropout_rate=0, supplement_mode=None):
        super(DenseGCNModel, self).__init__()
        print('DenseGCNModel Loaded')

        self.edge_dropout_rate = edge_dropout_rate
        self.num_layers = len(layers_dim) - 1
        self.graph_conv = DenseGCNBlock(layers_dim, 0.1, relu_layers_index=range(self.num_layers), dropout_layers_index=range(self.num_layers), supplement_mode=supplement_mode)

    def forward(self, graph, substitution_x=None, supplement_x=None):
        xs, adj, num_node1s, num_node2s = (substitution_x if substitution_x is not None else graph.x), graph.adj, graph.num_node1s, graph.num_node2s
        indexs = torch.where(adj != 0)
        edge_indexs = torch.cat((torch.unsqueeze(indexs[0], 0), torch.unsqueeze(indexs[1], 0)), 0)
        edge_indexs_dropout, edge_weights_dropout = dropout_adj(edge_index=edge_indexs, edge_attr=adj[indexs], p=self.edge_dropout_rate, force_undirected=True, num_nodes=num_node1s + num_node2s, training=self.training)
        adj_dropout = torch.zeros_like(adj)
        adj_dropout[edge_indexs_dropout[0], edge_indexs_dropout[1]] = edge_weights_dropout

        embeddings = self.graph_conv(xs, adj_dropout, supplement_x=supplement_x)

        return embeddings

class GATModel(torch.nn.Module):
    def __init__(self, layers_dim, supplement_mode=None, heads=1):
        super(GATModel, self).__init__()
        print('GATModel Loaded')

        self.num_layers = len(layers_dim) - 1
        self.graph_GAT = GATBlock(layers_dim, relu_layers_index=range(self.num_layers),supplement_mode=supplement_mode, heads=heads)

    def forward(self, graph_batchs, supplement_x=None,supplement_y=None):
        if supplement_x is not None:
            supplement_i = 0
            for graph_batch in graph_batchs:
                graph_batch.__setitem__('supplement_x', supplement_x[supplement_i: supplement_i + graph_batch.num_graphs])
                graph_batch.__setitem__('supplement_y',supplement_y)
                supplement_i += graph_batch.num_graphs

            embedding_batchs = list(map(lambda graph: self.graph_GAT(graph.x, graph.edge_index, None, graph.batch, supplement_x=graph.supplement_x[graph.batch.int().cpu().numpy()],supplement_y=graph.supplement_y[graph.batch.int().cpu().numpy()]), graph_batchs))
        else:
            embedding_batchs = list(map(lambda graph: self.graph_GAT(graph.x, graph.edge_index, None, graph.batch), graph_batchs))



        return embedding_batchs

class without_muti_model_GATModel(torch.nn.Module):
    def __init__(self, layers_dim, supplement_mode=None, heads=1):
        super(without_muti_model_GATModel, self).__init__()
        print('without_muti_model_GATModel Loaded')

        self.num_layers = len(layers_dim) - 1
        self.graph_GAT = without_muti_model_GATBlock(layers_dim, relu_layers_index=range(self.num_layers),supplement_mode=supplement_mode, heads=heads)

    def forward(self, graph_batchs, supplement_x=None):
        if supplement_x is not None:
            supplement_i = 0
            for graph_batch in graph_batchs:
                graph_batch.__setitem__('supplement_x', supplement_x[supplement_i: supplement_i + graph_batch.num_graphs])
                supplement_i += graph_batch.num_graphs
                embedding_batchs = list(map(lambda graph: self.graph_GAT(graph.x, graph.edge_index, None, graph.batch, supplement_x=graph.supplement_x[graph.batch.int().cpu().numpy()]), graph_batchs))
        else:
            embedding_batchs = list(map(lambda graph: self.graph_GAT(graph.x, graph.edge_index, None, graph.batch), graph_batchs))


        return embedding_batchs

class without_hete_net_GATModel(torch.nn.Module):
    def __init__(self, layers_dim, supplement_mode=None, heads=1):
        super(without_hete_net_GATModel, self).__init__()
        print('without_hete_net_GATModel Loaded')
        self.num_layers = len(layers_dim) - 1
        self.graph_GAT = without_hete_net_GATBlock(layers_dim, relu_layers_index=range(self.num_layers),supplement_mode=supplement_mode, heads=heads)

    def forward(self, graph_batchs, supplement_y=None):
        if supplement_y is not None:
            supplement_i = 0
            for graph_batch in graph_batchs:
                graph_batch.__setitem__('supplement_y', supplement_y)
                supplement_i += graph_batch.num_graphs
                embedding_batchs = list(map(lambda graph: self.graph_GAT(graph.x, graph.edge_index, None, graph.batch,supplement_y=graph.supplement_y[graph.batch.int().cpu().numpy()]),graph_batchs))
        else:
            embedding_batchs = list(map(lambda graph: self.graph_GAT(graph.x, graph.edge_index, None, graph.batch), graph_batchs))



        return embedding_batchs

class join_model_hete_GATModel(torch.nn.Module):
    def __init__(self, layers_dim, supplement_mode=None, heads=1):
        super(join_model_hete_GATModel, self).__init__()
        print('join_model_hete_GATModel Loaded')

        self.num_layers = len(layers_dim) - 1
        self.graph_GAT = join_model_hete_GATBlock(layers_dim, relu_layers_index=range(self.num_layers),supplement_mode=supplement_mode, heads=heads)

    def forward(self, graph_batchs, supplement_x=None,supplement_y=None):
        if supplement_x is not None:
            supplement_i = 0
            for graph_batch in graph_batchs:
                graph_batch.__setitem__('supplement_x', supplement_x[supplement_i: supplement_i + graph_batch.num_graphs])
                graph_batch.__setitem__('supplement_y',supplement_y)
                supplement_i += graph_batch.num_graphs

            embedding_batchs = list(map(lambda graph: self.graph_GAT(graph.x, graph.edge_index, None, graph.batch, supplement_x=graph.supplement_x[graph.batch.int().cpu().numpy()],supplement_y=graph.supplement_y[graph.batch.int().cpu().numpy()]), graph_batchs))
        else:
            embedding_batchs = list(map(lambda graph: self.graph_GAT(graph.x, graph.edge_index, None, graph.batch), graph_batchs))



        return embedding_batchs

class GCNModel(torch.nn.Module):
    def __init__(self, layers_dim, supplement_mode=None):
        super(GCNModel, self).__init__()
        print('GCNModel Loaded')

        self.num_layers = len(layers_dim) - 1
        self.graph_conv = GCNBlock(layers_dim, relu_layers_index=range(self.num_layers), supplement_mode=supplement_mode)

    def forward(self, graph_batchs, supplement_x=None):

        if supplement_x is not None:
            supplement_i = 0
            for graph_batch in graph_batchs:
                graph_batch.__setitem__('supplement_x', supplement_x[supplement_i: supplement_i + graph_batch.num_graphs])
                supplement_i += graph_batch.num_graphs
            embedding_batchs = list(map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch, supplement_x=graph.supplement_x[graph.batch.int().cpu().numpy()]), graph_batchs))
        else:
            embedding_batchs = list(map(lambda graph: self.graph_conv(graph.x, graph.edge_index, None, graph.batch), graph_batchs))

        embeddings = []
        for i in range(self.num_layers + 1):
            embeddings.append(torch.cat(list(map(lambda embedding_batch: embedding_batch[i], embedding_batchs)), 0))

        return embeddings

class ConvNet_or_GAT(torch.nn.Module):
    def __init__(self,ag_init_dim=2339, mg_init_dim=78, pg_init_dim=54, HGraph_dropout_rate=0.2, skip=False, embedding_dim=128, drug_smiles_init_dim=100,target_seq_init_dim=2560,integration_mode="combination4"):
        super(ConvNet_or_GAT, self).__init__()
        print('convNet_or_GAT Loaded')
        HGraph_dims=[ag_init_dim, 512, 256]

        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 2, mg_init_dim * 4,mg_init_dim * 4]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 2, pg_init_dim * 4, pg_init_dim * 4]

        drug_transform_dims = [HGraph_dims[-1], 1024, drug_graph_dims[1]]
        target_transform_dims = [HGraph_dims[-1], 1024, target_graph_dims[1]]


        drug_smiles_trans_dims=[drug_smiles_init_dim,1024,drug_graph_dims[2]]
        target_seq_trans_dim=[target_seq_init_dim,1024,target_graph_dims[2]]
        self.skip = skip
        if not skip:
            drug_output_dims = [drug_graph_dims[-1], 1024, embedding_dim]
            target_output_dims = [target_graph_dims[-1], 1024, embedding_dim]
        else:
            drug_output_dims = [drug_graph_dims[-1] + drug_transform_dims[-1], 1024, embedding_dim]
            target_output_dims = [target_graph_dims[-1] + target_transform_dims[-1], 1024, embedding_dim]

        self.output_dim = embedding_dim

        self.HGraph_Attention=DenseGATModel(HGraph_dims,HGraph_dropout_rate)
        self.drug_transform_linear = LinearBlock(drug_transform_dims, 0.1, relu_layers_index=[0],dropout_layers_index=[0, 1])
        self.target_transform_linear = LinearBlock(target_transform_dims, 0.1, relu_layers_index=[0], dropout_layers_index=[0, 1])

        self.drug_CNN=CNNBlock(65,64,drug_graph_dims[2])

        self.target_ESM2_transform_linear = LinearBlock(target_seq_trans_dim,0.1,relu_layers_index=[0],dropout_layers_index=[0, 1])

        self.drug_Graph_GAT=GATModel(drug_graph_dims,supplement_mode=integration_mode)
        self.target_Graph_GAT=GATModel(target_graph_dims,supplement_mode=integration_mode)

        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0],dropout_layers_index=[0, 1])
    def forward(self,HGraph,drug_graph_batchs,drug_smiles_embedding_batchs,target_graph_batchs,target_ESM2_embedding_baths,drug_map=None, drug_map_weight=None, target_map=None, target_map_weight=None):
        num_node1s, num_node2s =HGraph.num_node1s,HGraph.num_node2s
        HGraph_embedding=self.HGraph_Attention(HGraph)[0]

        if drug_map is not None:
            if drug_map_weight is not None:
                drug_transform_embedding = torch.sum(self.drug_transform_linear(HGraph_embedding[:num_node1s])[-1][drug_map, :] * drug_map_weight, dim=-2)
            else:
                drug_transform_embedding = torch.mean(self.drug_transform_linear(HGraph_embedding[:num_node1s])[-1][drug_map, :], dim=-2)
        else:
            drug_transform_embedding = self.drug_transform_linear(HGraph_embedding[:num_node1s])[0]

        if target_map is not None:
            if target_map_weight is not None:
                target_transform_embedding = torch.sum(self.target_transform_linear(HGraph_embedding[num_node1s:])[-1][target_map, :] * target_map_weight, dim=-2)
            else:
                target_transform_embedding = torch.mean(self.target_transform_linear(HGraph_embedding[num_node1s:])[-1][target_map, :], dim=-2)
        else:
            target_transform_embedding = self.target_transform_linear(HGraph_embedding[num_node1s:])[0]



        drug_smiles_embedding_batchs=drug_smiles_embedding_batchs.long()
        drug_smiles_transform_embedding=self.drug_CNN(drug_smiles_embedding_batchs)


        target_ESM2_transform_embedding=self.target_ESM2_transform_linear(target_ESM2_embedding_baths)[0]

        drug_graph_embedding=(self.drug_Graph_GAT(drug_graph_batchs,supplement_x=drug_transform_embedding,supplement_y=drug_smiles_transform_embedding)[0])[0]
        target_graph_embedding=(self.target_Graph_GAT(target_graph_batchs,supplement_x=target_transform_embedding,supplement_y=target_ESM2_transform_embedding)[0])[0]


        if not self.skip:
            drug_output_embedding = self.drug_output_linear(drug_graph_embedding)[0]
            target_output_embedding = self.target_output_linear(target_graph_embedding)[0]
        else:
            drug_output_embedding = self.drug_output_linear(torch.cat((drug_graph_embedding, drug_transform_embedding), 1))[0]
            target_output_embedding = self.target_output_linear(torch.cat((target_graph_embedding, target_transform_embedding), 1))[0]

        return drug_output_embedding, target_output_embedding

#HMH(without muti-modal)
class wo_muti_modal_ConvNet_or_GAT(torch.nn.Module):
    def __init__(self,ag_init_dim=2339, mg_init_dim=78, pg_init_dim=54, HGraph_dropout_rate=0.2, skip=False, embedding_dim=128, integration_mode="combination4"):
        super(wo_muti_modal_ConvNet_or_GAT, self).__init__()
        print('wo_muti_modal_ConvNet_or_GAT Loaded')
        HGraph_dims=[ag_init_dim, 512, 256]

        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 2, mg_init_dim * 4,mg_init_dim * 4]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 2, pg_init_dim * 4, pg_init_dim * 4]

        drug_transform_dims = [HGraph_dims[-1], 1024, drug_graph_dims[1]]
        target_transform_dims = [HGraph_dims[-1], 1024, target_graph_dims[1]]


        self.skip = skip
        if not skip:
            drug_output_dims = [drug_graph_dims[-1], 1024, embedding_dim]
            target_output_dims = [target_graph_dims[-1], 1024, embedding_dim]
        else:
            drug_output_dims = [drug_graph_dims[-1] + drug_transform_dims[-1], 1024, embedding_dim]
            target_output_dims = [target_graph_dims[-1] + target_transform_dims[-1], 1024, embedding_dim]

        self.output_dim = embedding_dim

        self.HGraph_Attention=DenseGATModel(HGraph_dims,HGraph_dropout_rate)
        self.drug_transform_linear = LinearBlock(drug_transform_dims, 0.1, relu_layers_index=[0],dropout_layers_index=[0, 1])
        self.target_transform_linear = LinearBlock(target_transform_dims, 0.1, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.drug_Graph_GAT=without_muti_model_GATModel(drug_graph_dims,supplement_mode=integration_mode)
        self.target_Graph_GAT=without_muti_model_GATModel(target_graph_dims,supplement_mode=integration_mode)

        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0],dropout_layers_index=[0, 1])
    def forward(self,HGraph,drug_graph_batchs,drug_smiles_embedding_batchs,target_graph_batchs,target_ESM2_embedding_baths,drug_map=None, drug_map_weight=None, target_map=None, target_map_weight=None):
        num_node1s, num_node2s =HGraph.num_node1s,HGraph.num_node2s
        HGraph_embedding=self.HGraph_Attention(HGraph)[0]

        if drug_map is not None:
            if drug_map_weight is not None:
                drug_transform_embedding = torch.sum(self.drug_transform_linear(HGraph_embedding[:num_node1s])[-1][drug_map, :] * drug_map_weight, dim=-2)
            else:
                drug_transform_embedding = torch.mean(self.drug_transform_linear(HGraph_embedding[:num_node1s])[-1][drug_map, :], dim=-2)
        else:
            drug_transform_embedding = self.drug_transform_linear(HGraph_embedding[:num_node1s])[0]

        if target_map is not None:
            if target_map_weight is not None:
                target_transform_embedding = torch.sum(self.target_transform_linear(HGraph_embedding[num_node1s:])[-1][target_map, :] * target_map_weight, dim=-2)
            else:
                target_transform_embedding = torch.mean(self.target_transform_linear(HGraph_embedding[num_node1s:])[-1][target_map, :], dim=-2)
        else:
            target_transform_embedding = self.target_transform_linear(HGraph_embedding[num_node1s:])[0]




        drug_graph_embedding=(self.drug_Graph_GAT(drug_graph_batchs,supplement_x=drug_transform_embedding)[0])[0]
        target_graph_embedding=(self.target_Graph_GAT(target_graph_batchs,supplement_x=target_transform_embedding)[0])[0]
        if not self.skip:
            drug_output_embedding = self.drug_output_linear(drug_graph_embedding)[0]
            target_output_embedding = self.target_output_linear(target_graph_embedding)[0]
        else:
            drug_output_embedding = self.drug_output_linear(torch.cat((drug_graph_embedding, drug_transform_embedding), 1))[0]
            target_output_embedding = self.target_output_linear(torch.cat((target_graph_embedding, target_transform_embedding), 1))[0]

        return drug_output_embedding, target_output_embedding

#HMH(without Hete-net)
class wo_Hete_net_ConvNet_or_GAT(torch.nn.Module):
    def __init__(self,ag_init_dim=2339, mg_init_dim=78, pg_init_dim=54, HGraph_dropout_rate=0.2, skip=False, embedding_dim=128, drug_smiles_init_dim=100,target_seq_init_dim=2560,integration_mode="combination4"):
        super(wo_Hete_net_ConvNet_or_GAT, self).__init__()
        print('wo_Hete_net_ConvNet_or_GAT Loaded')

        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 2, mg_init_dim * 4,mg_init_dim * 4]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 2, pg_init_dim * 4, pg_init_dim * 4]


        target_seq_trans_dim=[target_seq_init_dim,1024,target_graph_dims[2]]
        self.skip = skip
        if not skip:
            drug_output_dims = [drug_graph_dims[-1], 1024, embedding_dim]
            target_output_dims = [target_graph_dims[-1], 1024, embedding_dim]

        self.output_dim = embedding_dim


        self.drug_CNN=CNNBlock(65,64,drug_graph_dims[2])

        self.target_ESM2_transform_linear = LinearBlock(target_seq_trans_dim,0.1,relu_layers_index=[0],dropout_layers_index=[0, 1])

        self.drug_Graph_GAT=without_hete_net_GATModel(drug_graph_dims,supplement_mode=integration_mode)
        self.target_Graph_GAT=without_hete_net_GATModel(target_graph_dims,supplement_mode=integration_mode)

        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0],dropout_layers_index=[0, 1])
    def forward(self,HGraph,drug_graph_batchs,drug_smiles_embedding_batchs,target_graph_batchs,target_ESM2_embedding_baths,drug_map=None, drug_map_weight=None, target_map=None, target_map_weight=None):

        drug_smiles_embedding_batchs=drug_smiles_embedding_batchs.long()
        drug_smiles_transform_embedding=self.drug_CNN(drug_smiles_embedding_batchs)

        target_ESM2_transform_embedding=self.target_ESM2_transform_linear(target_ESM2_embedding_baths)[0]

        drug_graph_embedding=(self.drug_Graph_GAT(drug_graph_batchs,supplement_y=drug_smiles_transform_embedding)[0])[0]
        target_graph_embedding=(self.target_Graph_GAT(target_graph_batchs,supplement_y=target_ESM2_transform_embedding)[0])[0]
        if not self.skip:
            drug_output_embedding = self.drug_output_linear(drug_graph_embedding)[0]
            target_output_embedding = self.target_output_linear(target_graph_embedding)[0]

        return drug_output_embedding, target_output_embedding
#HMH(join_model_hete)
class join_model_hete_ConvNet_or_GAT(torch.nn.Module):
    def __init__(self,ag_init_dim=2339, mg_init_dim=78, pg_init_dim=54, HGraph_dropout_rate=0.2, skip=False, embedding_dim=128, drug_smiles_init_dim=100,target_seq_init_dim=2560,integration_mode="combination12"):
        super(join_model_hete_ConvNet_or_GAT, self).__init__()
        print('join_model_hete_ConvNet_or_GAT Loaded')
        HGraph_dims=[ag_init_dim, 512, 256]

        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 4, mg_init_dim * 4,mg_init_dim * 4]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 4, pg_init_dim * 4, pg_init_dim * 4]

        drug_transform_dims = [HGraph_dims[-1], 1024, drug_graph_dims[1]]
        target_transform_dims = [HGraph_dims[-1], 1024, target_graph_dims[1]]


        drug_smiles_trans_dims=[drug_smiles_init_dim,1024,drug_graph_dims[2]]
        target_seq_trans_dim=[target_seq_init_dim,1024,target_graph_dims[1]*2]
        self.skip = skip
        if not skip:
            drug_output_dims = [drug_graph_dims[-1], 1024, embedding_dim]
            target_output_dims = [target_graph_dims[-1], 1024, embedding_dim]
        else:
            drug_output_dims = [drug_graph_dims[-1] + drug_transform_dims[-1], 1024, embedding_dim]
            target_output_dims = [target_graph_dims[-1] + target_transform_dims[-1], 1024, embedding_dim]

        self.output_dim = embedding_dim

        self.HGraph_Attention=DenseGATModel(HGraph_dims,HGraph_dropout_rate)
        self.drug_transform_linear = LinearBlock(drug_transform_dims, 0.1, relu_layers_index=[0],dropout_layers_index=[0, 1])
        self.target_transform_linear = LinearBlock(target_transform_dims, 0.1, relu_layers_index=[0], dropout_layers_index=[0, 1])

        self.drug_CNN=CNNBlock(65,64,drug_graph_dims[1]*2)

        self.target_ESM2_transform_linear = LinearBlock(target_seq_trans_dim,0.1,relu_layers_index=[0],dropout_layers_index=[0, 1])

        self.drug_Graph_GAT=join_model_hete_GATModel(drug_graph_dims,supplement_mode=integration_mode)
        self.target_Graph_GAT=join_model_hete_GATModel(target_graph_dims,supplement_mode=integration_mode)

        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0],dropout_layers_index=[0, 1])
    def forward(self,HGraph,drug_graph_batchs,drug_smiles_embedding_batchs,target_graph_batchs,target_ESM2_embedding_baths,drug_map=None, drug_map_weight=None, target_map=None, target_map_weight=None):
        num_node1s, num_node2s =HGraph.num_node1s,HGraph.num_node2s
        HGraph_embedding=self.HGraph_Attention(HGraph)[0]

        if drug_map is not None:
            if drug_map_weight is not None:
                drug_transform_embedding = torch.sum(self.drug_transform_linear(HGraph_embedding[:num_node1s])[-1][drug_map, :] * drug_map_weight, dim=-2)
            else:
                drug_transform_embedding = torch.mean(self.drug_transform_linear(HGraph_embedding[:num_node1s])[-1][drug_map, :], dim=-2)
        else:
            drug_transform_embedding = self.drug_transform_linear(HGraph_embedding[:num_node1s])[0]

        if target_map is not None:
            if target_map_weight is not None:
                target_transform_embedding = torch.sum(self.target_transform_linear(HGraph_embedding[num_node1s:])[-1][target_map, :] * target_map_weight, dim=-2)
            else:
                target_transform_embedding = torch.mean(self.target_transform_linear(HGraph_embedding[num_node1s:])[-1][target_map, :], dim=-2)
        else:
            target_transform_embedding = self.target_transform_linear(HGraph_embedding[num_node1s:])[0]



        drug_smiles_embedding_batchs=drug_smiles_embedding_batchs.long()
        drug_smiles_transform_embedding=self.drug_CNN(drug_smiles_embedding_batchs)

        target_ESM2_transform_embedding=self.target_ESM2_transform_linear(target_ESM2_embedding_baths)[0]

        drug_graph_embedding=(self.drug_Graph_GAT(drug_graph_batchs,supplement_x=drug_transform_embedding,supplement_y=drug_smiles_transform_embedding)[0])[0]
        target_graph_embedding=(self.target_Graph_GAT(target_graph_batchs,supplement_x=target_transform_embedding,supplement_y=target_ESM2_transform_embedding)[0])[0]


        if not self.skip:
            drug_output_embedding = self.drug_output_linear(drug_graph_embedding)[0]
            target_output_embedding = self.target_output_linear(target_graph_embedding)[0]
        else:
            drug_output_embedding = self.drug_output_linear(torch.cat((drug_graph_embedding, drug_transform_embedding), 1))[0]
            target_output_embedding = self.target_output_linear(torch.cat((target_graph_embedding, target_transform_embedding), 1))[0]

        return drug_output_embedding, target_output_embedding

#HMH(target_rep_ESM)
class target_rep_ESM_ConvNet_or_GAT(torch.nn.Module):
    def __init__(self,ag_init_dim=2339, mg_init_dim=78, pg_init_dim=54, HGraph_dropout_rate=0.2, skip=False, embedding_dim=128, drug_smiles_init_dim=100,target_seq_init_dim=2560,integration_mode="combination4"):
        super(target_rep_ESM_ConvNet_or_GAT, self).__init__()
        print('target_rep_ESM_ConvNet_or_GAT Loaded')
        HGraph_dims=[ag_init_dim, 512, 256]

        drug_graph_dims = [mg_init_dim, mg_init_dim, mg_init_dim * 2, mg_init_dim * 4,mg_init_dim * 4]
        target_graph_dims = [pg_init_dim, pg_init_dim, pg_init_dim * 2, pg_init_dim * 4, pg_init_dim * 4]

        drug_transform_dims = [HGraph_dims[-1], 1024, drug_graph_dims[1]]
        target_transform_dims = [HGraph_dims[-1], 1024, target_graph_dims[1]]


        drug_smiles_trans_dims=[drug_smiles_init_dim,1024,drug_graph_dims[2]]
        target_seq_trans_dim=[target_seq_init_dim,1024,target_graph_dims[2]]
        self.skip = skip
        if not skip:
            drug_output_dims = [drug_graph_dims[-1], 1024, embedding_dim]
            target_output_dims = [target_graph_dims[-1], 1024, embedding_dim]
        else:
            drug_output_dims = [drug_graph_dims[-1] + drug_transform_dims[-1], 1024, embedding_dim]
            target_output_dims = [target_graph_dims[-1] + target_transform_dims[-1], 1024, embedding_dim]

        self.output_dim = embedding_dim

        self.HGraph_Attention=DenseGATModel(HGraph_dims,HGraph_dropout_rate)
        self.drug_transform_linear = LinearBlock(drug_transform_dims, 0.1, relu_layers_index=[0],dropout_layers_index=[0, 1])
        self.target_transform_linear = LinearBlock(target_transform_dims, 0.1, relu_layers_index=[0], dropout_layers_index=[0, 1])

        self.drug_CNN=CNNBlock(65,64,drug_graph_dims[2])
        self.target_CNN=CNNBlock_target(26,64,target_graph_dims[2])

        self.drug_Graph_GAT=GATModel(drug_graph_dims,supplement_mode=integration_mode)
        self.target_Graph_GAT=GATModel(target_graph_dims,supplement_mode=integration_mode)

        self.drug_output_linear = LinearBlock(drug_output_dims, 0.2, relu_layers_index=[0], dropout_layers_index=[0, 1])
        self.target_output_linear = LinearBlock(target_output_dims, 0.2, relu_layers_index=[0],dropout_layers_index=[0, 1])
    def forward(self,HGraph,drug_graph_batchs,drug_smiles_embedding_batchs,target_graph_batchs,target_seq_embedding_baths,drug_map=None, drug_map_weight=None, target_map=None, target_map_weight=None):
        num_node1s, num_node2s =HGraph.num_node1s,HGraph.num_node2s
        HGraph_embedding=self.HGraph_Attention(HGraph)[0]

        if drug_map is not None:
            if drug_map_weight is not None:
                drug_transform_embedding = torch.sum(self.drug_transform_linear(HGraph_embedding[:num_node1s])[-1][drug_map, :] * drug_map_weight, dim=-2)
            else:
                drug_transform_embedding = torch.mean(self.drug_transform_linear(HGraph_embedding[:num_node1s])[-1][drug_map, :], dim=-2)
        else:
            drug_transform_embedding = self.drug_transform_linear(HGraph_embedding[:num_node1s])[0]

        if target_map is not None:
            if target_map_weight is not None:
                target_transform_embedding = torch.sum(self.target_transform_linear(HGraph_embedding[num_node1s:])[-1][target_map, :] * target_map_weight, dim=-2)
            else:
                target_transform_embedding = torch.mean(self.target_transform_linear(HGraph_embedding[num_node1s:])[-1][target_map, :], dim=-2)
        else:
            target_transform_embedding = self.target_transform_linear(HGraph_embedding[num_node1s:])[0]



        drug_smiles_embedding_batchs=drug_smiles_embedding_batchs.long()
        drug_smiles_transform_embedding=self.drug_CNN(drug_smiles_embedding_batchs)
        target_seq_embedding_baths=target_seq_embedding_baths.long()
        target_seq_transform_embedding=self.target_CNN(target_seq_embedding_baths)


        drug_graph_embedding=(self.drug_Graph_GAT(drug_graph_batchs,supplement_x=drug_transform_embedding,supplement_y=drug_smiles_transform_embedding)[0])[0]
        target_graph_embedding=(self.target_Graph_GAT(target_graph_batchs,supplement_x=target_transform_embedding,supplement_y=target_seq_transform_embedding)[0])[0]


        if not self.skip:
            drug_output_embedding = self.drug_output_linear(drug_graph_embedding)[0]
            target_output_embedding = self.target_output_linear(target_graph_embedding)[0]
        else:
            drug_output_embedding = self.drug_output_linear(torch.cat((drug_graph_embedding, drug_transform_embedding), 1))[0]
            target_output_embedding = self.target_output_linear(torch.cat((target_graph_embedding, target_transform_embedding), 1))[0]

        return drug_output_embedding, target_output_embedding



class Predictor(torch.nn.Module):
    def __init__(self, embedding_dim=128, output_dim=2, prediction_mode="cat"):
        super(Predictor, self).__init__()
        print('Predictor Loaded')

        self.prediction_func, prediction_dim_func = vector_operations[prediction_mode]
        mlp_layers_dim = [prediction_dim_func(embedding_dim), 1024, 512, output_dim]

        self.mlp = LinearBlock(mlp_layers_dim, 0.1, relu_layers_index=[0, 1], dropout_layers_index=[0, 1])

    def forward(self, data, drug_embedding, target_embedding,flag):

        drug_id, target_id, y = data.drug_id, data.target_id, data.y
        if flag==True:
            concatenated_embedding = torch.cat((drug_embedding, target_embedding), dim=0)
            drug_feature_have_target=concatenated_embedding[drug_id.int().cpu().numpy()]
            target_feature_have_drug=concatenated_embedding[target_id.int().cpu().numpy()]
            concat_feature = self.prediction_func(drug_feature_have_target, target_feature_have_drug)
        else:
            drug_feature = drug_embedding[drug_id.int().cpu().numpy()]
            target_feature = target_embedding[target_id.int().cpu().numpy()]
            concat_feature = self.prediction_func(drug_feature, target_feature)



        mlp_embeddings = self.mlp(concat_feature)

        out = mlp_embeddings[0]

        return out
