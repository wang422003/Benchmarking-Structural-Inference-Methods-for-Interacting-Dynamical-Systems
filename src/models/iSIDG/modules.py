import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
from utils import my_softmax, gumbel_softmax, inter_check_
# torch.set_default_tensor_type('torch.DoubleTensor')


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""
    """Based on https://github.com/ethanfetaya/NRI"""
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        # print("====input of batchnorm: {}".format(inputs.size()))
        x = inputs.view(inputs.size()[0] * inputs.size()[1], -1)
        # print("====input after view: {}".format(x.size()))
        x = self.bn(x)
        return x.view(inputs.size()[0], inputs.size()[1], -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class MLPEdges(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0., max_pool=True):
        super(MLPEdges, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        if max_pool:
            self.pool = nn.MaxPool1d(2)
        else:
            self.pool = nn.AvgPool1d(2)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        # print("====input of batchnorm: {}".format(inputs.size()))
        x = inputs.view(inputs.size()[0] * inputs.size()[1], -1)
        # print("====input after view: {}".format(x.size()))
        x = self.bn(x)
        return x.view(inputs.size()[0], inputs.size()[1], -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        x = self.pool(x)
        return self.batch_norm(x)


class MLP_GRU(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP_GRU, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        # print("====input of batchnorm: {}".format(inputs.size()))
        size_m = int(inputs.nelement() / inputs.size()[-1])
        x = inputs.view(size_m, -1)
        # print("====input after view: {}".format(x.size()))
        x = self.bn(x)
        return x.view(inputs.size()[0], inputs.size()[1], -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        # print("input of mlp5 sizeL {}".format(inputs.size()))
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class MLPfGRU(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLPfGRU, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # def batch_norm(self, inputs):
    #     x = inputs.view(inputs.size(0) * inputs.size(1), -1)
    #     x = self.bn(x)
    #     return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.bn(x)


class GRUCell(nn.Module):
    """
    An implementation of GRUCell.
    Customized GRU module,
    refer to 'Neural Relational Inference with Efficient Message Passing Mechanism'

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        print("-" * 30)
        print("x size")
        print(x.size())
        # x = x.view(-1, x.size(1))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        print("gate_x:")
        print(gate_x.size())
        gate_h = gate_h.squeeze()
        print("gate_h:")
        print(gate_h.size())

        i_r, i_i, i_n = gate_x.chunk(3, 2)
        h_r, h_i, h_n = gate_h.chunk(3, 2)

        resetgate = F.sigmoid(i_r + h_r)
        inputgate = F.sigmoid(i_i + h_i)
        newgate = F.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)

        return hy


class GRUModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRUModule, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        self.gru_cell = GRUCell(input_dim, hidden_dim, layer_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        # print(x.shape,"x.shape")100, 28, 28
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), x.size(1), self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), x.size(1), self.hidden_dim))

        outs = []

        hn = h0[0, :, :, :]

        for seq in range(x.size(1)):
            hn = self.gru_cell(x[:, seq, :, :], hn)
            outs.append(hn)

        out = outs[-1].squeeze()

        # out = self.fc(out)
        # out.size() --> 100, 10
        return out


class CNN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                                 dilation=1, return_indices=False,
                                 ceil_mode=False)

        self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(n_hid, 1, kernel_size=1)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        # Input shape: [num_sims * num_edges, num_dims, num_timesteps]

        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        pred = self.conv_predict(x)
        attention = my_softmax(self.conv_attention(x), axis=2)

        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob


class MPMEncoder(nn.Module):
    """Based on https://github.com/ethanfetaya/NRI"""
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True, output_tensor=False, gin=False):
        super(MPMEncoder, self).__init__()

        self.factor = factor
        self.gin = gin
        self.output_tensor_flag = output_tensor  # True -> output the tensor after mlp4 as well
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size()[1]

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send, adj):
        # Input shape: [num_sims, num_nodes, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_nodes, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node
        node_skip = x
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            if self.gin:
                x = x + node_skip
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        out = self.fc_out(x)
        return out


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder
    """
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True, output_tensor=False):
        super(MLPEncoder, self).__init__()
        self.factor = factor
        self.output_tensor_flag = output_tensor  # True -> output the tensor after mlp4 as well
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid * 2, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size()[1]

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send, adj):
        # Input shape: [num_sims, num_nodes, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_nodes, num_timesteps*num_dims]
        x = self.mlp1(x)  # 2-layer ELU net per node
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x = self.mlp3(x)
        out = self.fc_out(x)
        return out


class GINEncoder(nn.Module):
    """
    Modified MLP encoder with invariance and other stuffs.
    """
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True, gin=True):
        super(GINEncoder, self).__init__()

        self.factor = factor
        self.gin = gin
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLPEdges(n_hid * 2, n_hid * 2, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLPEdges(n_hid * 3, n_hid * 2, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size()[1]

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send, adj):
        # Input shape: [num_sims, num_nodes, num_timesteps, num_dims]
        x = inputs.view(inputs.size()[0], inputs.size()[1], -1)
        # New shape: [num_sims, num_nodes, num_timesteps*num_dims]
        # print("x")
        # print(x.size())
        x = self.mlp1(x)  # 2-layer ELU net per node
        node_skip = x
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            if self.gin:
                x = x + node_skip
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)


class GCNEncoder(nn.Module):

    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(GCNEncoder, self).__init__()

        self.factor = factor
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLPEdges(n_hid * 2, n_hid * 2, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.gcn = GraphConvolution(n_hid, n_hid)
        if self.factor:
            self.mlp4 = MLPEdges(n_hid * 3, n_hid * 2, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size()[1]

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send, adj):
        # Input shape: [num_sims, num_nodes, num_timesteps, num_dims]
        x = inputs.view(inputs.size()[0], inputs.size()[1], -1)
        # New shape: [num_sims, num_nodes, num_timesteps*num_dims]
        # print("x")
        # print(x.size())
        x = self.mlp1(x)  # 2-layer ELU net per node
        x_skip = self.node2edge(x, rel_rec, rel_send)
        x_skip = self.mlp2(x_skip)

        if self.factor:
            x = self.gcn(x, adj)
            # x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)


class GATEncoder(nn.Module):

    def __init__(self, n_feat, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(GATEncoder, self).__init__()

        self.factor = factor
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLPEdges(n_hid * 2, n_hid * 2, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.gat = GraphAttentionLayer(n_hid, n_hid, 1-do_prob, alpha=0.2, concat=True)
        if self.factor:
            self.mlp4 = MLPEdges(n_hid * 3, n_hid * 2, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size()[1]

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send, adj):
        # Input shape: [num_sims, num_nodes, num_timesteps, num_dims]
        x = inputs.view(inputs.size()[0], inputs.size()[1], -1)
        # New shape: [num_sims, num_nodes, num_timesteps*num_dims]
        # print("x")
        # print(x.size())
        x = self.mlp1(x)  # 2-layer ELU net per node
        x_skip = self.node2edge(x, rel_rec, rel_send)
        x_skip = self.mlp2(x_skip)

        if self.factor:
            x = self.gat(x, adj)
            # x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)


class GRUEncoder(nn.Module):
    """
    GEU encoder with spatio-temporal feature concatenation
    """
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True, cuda=True):
        super(GRUEncoder, self).__init__()

        self.factor = factor
        if cuda and torch.cuda.is_available():
            self.cuda_ = True
        else:
            self.cuda_ = False

        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.mlp5 = MLP_GRU(n_hid * 2, n_hid, n_hid, do_prob)
        self.intraedge = nn.GRU(n_hid, n_hid)
        self.interedge = nn.GRU(n_hid, n_hid)
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()
        if self.cuda_:
            self.mlp1 = self.mlp1.cuda()
            self.mlp2 = self.mlp2.cuda()
            self.mlp3 = self.mlp3.cuda()
            self.mlp4 = self.mlp4.cuda()
            self.mlp5 = self.mlp5.cuda()
            self.intraedge = self.intraedge.cuda()
            self.interedge = self.interedge.cuda()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size()[1]

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_nodes, num_timesteps, num_dims]
        # print("Size input: {}".format(inputs.size()))
        # print("rel_rec size: {}".format(rel_rec.size()))
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_nodes, num_timesteps*num_dims]
        # print("Size input after view method: {}".format(x.size()))

        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            # print("factor")
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        # inter edge operations
        # print("Size input after mlp4: {}".format(x.size()))  # ([128, 49, 512])

        x_exp = x.view(x.size()[0], rel_rec.size()[1], int(x.size()[1] / rel_rec.size()[1]), -1)
        # if self.cuda:
        #     x_exp = x_exp.cuda()
        x_exp = x_exp.permute(1, 2, 0, -1)  # shape: (num_nodes, num_nodes, num_batch, 256)
        # print("Size input after x_exp after permute: {}".format(x_exp.size()))  # ([7, 7, 128, 512])
        # springs: torch.Size([5, 4, 128, 256])

        x_concat_up = torch.empty((rel_rec.size()[1], int(x.size()[1] / rel_rec.size()[1]), x_exp.size()[2], x_exp.size()[-1]))
        # print("Size input after x_concat_up after permute: {}".format(x_concat_up.size()))  # ([7, 7, 512, 512])
        if self.cuda_:
            x_concat_up = x_concat_up.cuda()
        # x_concat_down = torch.empty((rel_rec.size(0), x_exp.size()[3], x_exp.size()[-1]))
        for i in range(x_exp.size()[1]):
            flag = False
            if int(x.size()[1] / rel_rec.size()[1]) != rel_rec.size()[1]:
                x_extract_up = 0
                flag = True
            else:
                x_extract_up = x_exp[i][i]
                x_extract_up = x_extract_up.unsqueeze(0)
            # else:
            #     x_extract_up = x_exp[i][x_exp.size()[1] - 1]
            j = i + 1
            while j < x_exp.size()[0]:
                if flag:
                    x_extract_up = x_exp[j][i].unsqueeze(0)
                    j += 1
                    continue
                x_extract_up = torch.cat((x_extract_up, x_exp[j][i].unsqueeze(0)), dim=0)
                j += 1
            if i != 0:
                p = 0
                while p < i:
                    if flag:
                        x_extract_up = x_exp[p][i].unsqueeze(0)
                        p += 1
                        continue
                    x_extract_up = torch.cat((x_extract_up, x_exp[p][i].unsqueeze(0)), dim=0)
                    p += 1
            x_concat_up[i] = x_extract_up
        x_edges_1 = torch.empty(rel_rec.size()[1], int(x.size()[1] / rel_rec.size()[1]), x_exp.size()[2], x_exp.size()[-1])
        # if self.cuda:
        #     x_edges_1 = x_edges_1.cuda()
        h0 = torch.zeros_like(x_concat_up[0][0])
        h0 = h0.unsqueeze(0)
        if self.cuda_:
            h0 = h0.cuda()

        for ind in range(len(x_concat_up)):
            e1, hn = self.intraedge(x_concat_up[ind], h0)
            x_edges_1[ind] = e1
        x_edges = [torch.mean(entity, keepdim=True, dim=0) for entity in x_edges_1]
        x_edges = torch.cat(x_edges, dim=0)
        # print("x_edges: {}".format(x_edges.size()))  # springs: torch.Size([5, 128, 256])
        # if self.cuda:
        #     x_edges = x_edges.cuda()
        # todo: check the dimenstion from this point
        x_concat_2 = torch.empty((rel_rec.size()[1], int(x.size()[1] / rel_rec.size()[1]), x_edges.size()[1], x_edges.size()[-1]))
        if self.cuda_:
            x_concat_2 = x_concat_2 .cuda()
        # x_concat_down = torch.empty((rel_rec.size(0), x_exp.size()[3], x_exp.size()[-1]))
        for i in range(x_edges.size()[0]):
            flag = False
            if int(x.size()[1] / rel_rec.size()[1]) != rel_rec.size()[1]:
                x_extract_2 = 0
                flag = True
            else:
                x_extract_2 = x_edges[i]
                x_extract_2 = x_extract_2.unsqueeze(0)
            j = i + 1
            while j < x_exp.size()[0]:
                if flag:
                    x_extract_2 = x_exp[j].unsqueeze(0)
                    j += 1
                    continue
                x_extract_2 = torch.cat((x_extract_2, x_edges[j].unsqueeze(0)), dim=0)
                j += 1
            if i != 0:
                p = 0
                while p < i:
                    if flag:
                        x_extract_2 = x_exp[p].unsqueeze(0)
                        p += 1
                        continue
                    x_extract_2 = torch.cat((x_extract_2, x_edges[p].unsqueeze(0)), dim=0)
                    p += 1
            x_concat_2[i] = x_extract_2
        x_edges_2 = torch.empty(rel_rec.size()[1], int(x.size()[1] / rel_rec.size()[1]), x_edges.size()[1], x_edges.size()[-1])
        # print("x_edges_2: {}".format(x_edges_2.size()))  # springs: torch.Size([5, 5, 128, 256])
        # if self.cuda:
        #     x_edges_2 = x_edges_2.cuda()
        h0_2 = torch.zeros_like(x_concat_2[0][0])
        h0_2 = h0_2.unsqueeze(0)
        if self.cuda_:
            h0_2 = h0_2.cuda()
        for ind in range(len(x_concat_2)):
            e1, hn = self.interedge(x_concat_2[ind], h0_2)
            x_edges_2[ind] = e1
        # print("x_edges_1: {}".format(x_edges_1.size()))  # springs: torch.Size([5, 5, 128, 256])
        # print("x_edges_2: {}".format(x_edges_2.size()))  # springs: torch.Size([5, 5, 128, 256])
        x_1 = torch.cat((x_edges_1, x_edges_2), dim=-1)
        # print("x_1: {}".format(x_1.size()))
        if self.cuda_:
            x_1 = x_1.cuda()
        x_1 = x_1.view(x_1.size()[0] * x_1.size()[1], x_1.size()[2], -1)
        x_1 = x_1.permute(1, 0, 2)
        # print("x_1 new: {}".format(x_1.size()))
        x = self.mlp5(x_1)
        # print("x after mlp5: {}".format(x.size()))  # springs: torch.Size([128, 25, 256])
        # print("-" * 30)
        return self.fc_out(x)


class CNNEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(CNNEncoder, self).__init__()
        self.dropout_prob = do_prob

        self.factor = factor

        self.cnn = CNN(n_in * 2, n_hid, n_hid, do_prob)
        self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)

        if self.factor:
            print("Using factor graph CNN encoder.")
        else:
            print("Using CNN encoder.")

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.

        x = inputs.view(inputs.size()[0], inputs.size()[1] -1)

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.view(inputs.size()[0] * receivers.size()[1],
                                   inputs.size()[2], inputs.size()[3])
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.view(inputs.size()[0] * senders.size()[1],
                               inputs.size()[2],
                               inputs.size()[3])
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([senders, receivers], dim=1)
        return edges

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size()[1]

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):

        # Input has shape: [num_sims, num_nodes, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        x = self.cnn(edges)
        x = x.view(inputs.size(0), (inputs.size(1) - 1) * inputs.size(1), -1)
        x = self.mlp1(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp2(x)

            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp3(x)

        return self.fc_out(x)


class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False):
        super(MLPDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_nodes, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_nodes*(num_nodes-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size()[0], pre_msg.size()[1],
                                        pre_msg.size()[2], self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exclude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            # print("msg: {}".format(msg.size()))
            # print("single_timestep_rel_type: {}".format(single_timestep_rel_type.size()))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                 rel_type.size(2)]
        rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run 1 prediction step
        last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                             curr_rel_type)
        preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()


class MULTIDecoder(nn.Module):
    """MULTI decoder module. From https://github.com/ethanfetaya/NRI/blob/master/modules.py"""

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False):
        super(MULTIDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_nodes, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_nodes*(num_nodes-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size()[0], pre_msg.size()[1],
                                        pre_msg.size()[2], self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exclude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                 rel_type.size(2)]
        rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                 curr_rel_type)
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()


class RNNDecoder(nn.Module):
    """Recurrent decoder module. From https://github.com/ethanfetaya/NRI/blob/master/modules.py"""

    def __init__(self, n_in_node, edge_types, n_hid,
                 do_prob=0., skip_first=False):
        super(RNNDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.hidden_r = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_i = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_h = nn.Linear(n_hid, n_hid, bias=False)

        self.input_r = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_i = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_n = nn.Linear(n_in_node, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, inputs, rel_rec, rel_send,
                            rel_type, hidden):

        # node2edge
        receivers = torch.matmul(rel_rec, hidden)
        senders = torch.matmul(rel_send, hidden)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size()[0], pre_msg.size()[1],
                                        self.msg_out_shape))
        if inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2)) - 1.
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2))

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.tanh(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg / norm

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2,
                                                                        -1)
        agg_msgs = agg_msgs.contiguous() / inputs.size(2)  # Average

        # GRU-style gated aggregation
        r = F.sigmoid(self.input_r(inputs) + self.hidden_r(agg_msgs))
        i = F.sigmoid(self.input_i(inputs) + self.hidden_i(agg_msgs))
        n = F.tanh(self.input_n(inputs) + r * self.hidden_h(agg_msgs))
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        pred = inputs + pred

        return pred, hidden

    def forward(self, data, rel_type, rel_rec, rel_send, pred_steps=1,
                burn_in=False, burn_in_steps=1, dynamic_graph=False,
                encoder=None, temp=None):

        inputs = data.transpose(1, 2).contiguous()

        time_steps = inputs.size(1)

        # inputs has shape
        # [batch_size, num_timesteps, num_nodes, num_dims]

        # rel_type has shape:
        # [batch_size, num_nodes*(num_nodes-1), num_edge_types]

        hidden = Variable(
            torch.zeros(inputs.size(0), inputs.size(2), self.msg_out_shape))
        if inputs.is_cuda:
            hidden = hidden.cuda()

        pred_all = []

        for step in range(0, inputs.size(1) - 1):

            if burn_in:
                if step <= burn_in_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]
            else:
                assert (pred_steps <= time_steps)
                # Use ground truth trajectory input vs. last prediction
                if not step % pred_steps:
                    ins = inputs[:, step, :, :]
                else:
                    ins = pred_all[step - 1]

            if dynamic_graph and step >= burn_in_steps:
                # NOTE: Assumes burn_in_steps = args.timesteps
                logits = encoder(
                    data[:, :, step - burn_in_steps:step, :].contiguous(),
                    rel_rec, rel_send)
                rel_type = gumbel_softmax(logits, tau=temp, hard=True)

            pred, hidden = self.single_step_forward(ins, rel_rec, rel_send,
                                                    rel_type, hidden)
            pred_all.append(pred)

        preds = torch.stack(pred_all, dim=1)

        return preds.transpose(1, 2).contiguous()


class GRUDecoder(nn.Module):
    """
    GRU decoder module.
    Based on the paper of Neural Relational Inference with Efficient Message Passing Mechanism
    """

    def __init__(self, n_in_node, edge_types, n_hid, do_prob=0., skip_first=False):
        super(GRUDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, n_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.mlp_n2e = MLP(n_in=n_hid, n_hid=n_hid, n_out=n_hid, do_prob=do_prob)
        self.mlp_e2n = MLP(n_in=n_hid + n_in_node, n_hid=n_hid + n_in_node,
                           n_out=n_hid + n_in_node, do_prob=do_prob)

        self.hidden_er = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_ei = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_eh = nn.Linear(n_hid, n_hid, bias=False)

        self.input_er = nn.Linear(n_hid, n_hid, bias=True)
        self.input_ei = nn.Linear(n_hid, n_hid, bias=True)
        self.input_en = nn.Linear(n_hid, n_hid, bias=True)

        self.hidden_nr = nn.Linear(n_hid + n_in_node, n_hid + n_in_node, bias=False)
        self.hidden_ni = nn.Linear(n_hid + n_in_node, n_hid + n_in_node, bias=False)
        self.hidden_nh = nn.Linear(n_hid + n_in_node, n_hid + n_in_node, bias=False)

        self.input_nr = nn.Linear(n_hid + n_in_node, n_hid + n_in_node, bias=True)
        self.input_ni = nn.Linear(n_hid + n_in_node, n_hid + n_in_node, bias=True)
        self.input_nn = nn.Linear(n_hid + n_in_node, n_hid + n_in_node, bias=True)

        self.out_fc1 = nn.Linear(n_in_node + n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)
        self.n_in_node = n_in_node
        self.n_hid = n_hid

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward_ini(self, single_timestep_inputs, rel_rec, rel_send,
                                single_timestep_rel_type):
        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_nodes, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_nodes*(num_nodes-1), num_edge_types]

        # Node2edge
        # receivers = torch.matmul(rel_rec.double(), single_timestep_inputs.double())  # torch.Size([100, 5, 49, 1])
        # senders = torch.matmul(rel_send.double(), single_timestep_inputs.double())  # torch.Size([100, 5, 49, 1])
        # pre_msg = torch.cat([senders, receivers], dim=-1).double()  # torch.Size([100, 5, 49, 2])

        receivers = torch.matmul(rel_rec, single_timestep_inputs)  # torch.Size([100, 5, 49, 1])
        senders = torch.matmul(rel_send, single_timestep_inputs)  # torch.Size([100, 5, 49, 1])
        pre_msg = torch.cat([senders, receivers], dim=-1)  # torch.Size([100, 5, 49, 2])

        all_msgs = Variable(torch.zeros(pre_msg.size()[0], pre_msg.size()[1],
                                        pre_msg.size()[2], self.msg_out_shape))  # torch.Size([100, 5, 49, 256])

        # print(single_timestep_rel_type.size())  # torch.Size([100, 5, 49, 2])
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()
        start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))  # .double()
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg
        # print("all_msgs:")  # torch.Size([100, 5, 49, 256])  # (num_batch, num_timesteps, num_edges, num_features)
        # print(all_msgs.size())  # torch.Size([100, 5, 49, 256])

        # Aggregate all msgs to receiver
        h_edges_1 = [self.mlp_n2e(all_msgs[:, i, :, :]).unsqueeze(1) for i in
                     range(all_msgs.size()[1])]  # size: (100, 49, 256)
        # print(h_edges_1[0].size())  # torch.Size([100, 1, 49, 256])
        # print(len(h_edges_1))  # 5
        h_edges_1 = torch.cat(h_edges_1, dim=1)

        # print(h_edges_1.size())  #  torch.Size([100, 5, 49, 256])

        # agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        # agg_msgs = agg_msgs.contiguous()  # torch.Size([100, 5, 49, 256])
        agg_msgs = h_edges_1.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        # agg_msgs = h_edges_1.transpose(-2, -1).matmul(rel_rec.double()).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()  # torch.Size([100, 5, 7, 256])

        # Skip connection
        aug_inputs = torch.cat([agg_msgs, single_timestep_inputs], dim=-1)  # torch.Size([100, 5, 7, 257])
        # print("aug_inputs:")  # torch.Size([100, 5, 7, 257])
        # print(aug_inputs.size())
        h_nodes_1 = [self.mlp_e2n(aug_inputs[:, i, :, :]).unsqueeze(1) for i in range(aug_inputs.size()[1])]
        h_nodes_1 = torch.cat(h_nodes_1, dim=1)
        # print("h_nodes_1:")  # torch.Size([100, 5, 7, 257])
        # print(h_nodes_1.size())  # torch.Size([100, 5, 7, 257])

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(h_nodes_1)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred, h_edges_1, h_nodes_1

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type, hidden_n2e, hidden_e2n, last_step_flag=False):
        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_nodes, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_nodes*(num_nodes-1), num_edge_types]
        if last_step_flag:
            size = single_timestep_inputs.size()[1]
            hidden_n2e = hidden_n2e[:, :size, :, :]
            hidden_e2n = hidden_e2n[:, :size, :, :]
        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)  # torch.Size([100, 5, 49, 1])
        senders = torch.matmul(rel_send, single_timestep_inputs)  # torch.Size([100, 5, 49, 1])
        pre_msg = torch.cat([senders, receivers], dim=-1)  # torch.Size([100, 5, 49, 2])

        # receivers = torch.matmul(rel_rec.double(), single_timestep_inputs.double())  # torch.Size([100, 5, 49, 1])
        # senders = torch.matmul(rel_send.double(), single_timestep_inputs.double())  # torch.Size([100, 5, 49, 1])
        # pre_msg = torch.cat([senders, receivers], dim=-1)  # torch.Size([100, 5, 49, 2])

        all_msgs = Variable(torch.zeros(pre_msg.size()[0], pre_msg.size()[1],
                                        pre_msg.size()[2], self.msg_out_shape))

        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()
        start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))  # pre_msg.double()
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg
        # print("all_msgs:")  # torch.Size([100, 5, 49, 256])  # (num_batch, num_timesteps, num_edges, num_features)

        # GRU-style gated aggregation

        er = F.sigmoid(self.input_er(all_msgs) + self.hidden_er(hidden_n2e))  # torch.Size([100, 5, 49, 256])

        ei = F.sigmoid(self.input_ei(all_msgs) + self.hidden_ei(hidden_n2e))  # torch.Size([100, 5, 49, 256])

        en = F.tanh(self.input_en(all_msgs) + er * self.hidden_eh(hidden_n2e))  # torch.Size([100, 5, 49, 256])

        # print(hidden_n2e.size())   # torch.Size([100, 5, 49, 256])
        h_edges_1 = (1 - ei) * en + ei * hidden_n2e  # torch.Size([100, 5, 49, 256])

        agg_msgs = h_edges_1.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()  # torch.Size([100, 5, 7, 256])

        # Skip connection
        aug_inputs = torch.cat([agg_msgs, single_timestep_inputs], dim=-1)  # torch.Size([100, 5, 7, 257])

        nr = F.sigmoid(self.input_nr(aug_inputs) + self.hidden_nr(hidden_e2n))  # torch.Size([100, 5, 7, 257])

        ni = F.sigmoid(self.input_ni(aug_inputs) + self.hidden_ni(hidden_e2n))  # torch.Size([100, 5, 7, 257])

        nn_ = F.tanh(self.input_nn(aug_inputs) + nr * self.hidden_nh(hidden_e2n))  # torch.Size([100, 5, 7, 257])
        # print(hidden_e2n.size())   # torch.Size([100, 5, 7, 257])
        h_nodes_1 = (1 - ni) * nn_ + ni * hidden_e2n  # torch.Size([100, 5, 7, 257])

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(h_nodes_1)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred, h_edges_1, h_nodes_1

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [rel_type.size()[0], inputs.size()[1], rel_type.size()[1],
                 rel_type.size()[2]]
        # print("sizes: {}".format(sizes))
        rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        first_step = inputs[:, 0::pred_steps, :, :]  # torch.Size([100, 5, 7, 1])
        # first_step = first_step.double()
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).
        hidden_n2e = Variable(torch.zeros(inputs.size()[0], first_step.size()[1], rel_type.size()[2],
                                          self.n_hid))  # torch.Size([100, 5, 49, 256])
        hidden_e2n = Variable(torch.zeros(inputs.size()[0], first_step.size()[1], first_step.size()[2],
                                          self.n_hid + self.n_in_node))  # [100, 5, 7, 257]
        # Run n prediction steps
        for step in range(0, pred_steps):

            if step == 0:
                # print(step)
                last_pred, hidden_n2e, hidden_e2n = self.single_step_forward_ini(first_step, rel_rec, rel_send,
                                                                                 curr_rel_type)
            else:
                # print(step)
                this_step = inputs[:, step::pred_steps, :, :]  # torch.Size([100, 5, 7, 1])
                this_rel_type = rel_type[:, step::pred_steps, :, :]
                flag = first_step.size()[1] > this_step.size()[1]
                last_pred, hidden_n2e, hidden_e2n = self.single_step_forward(this_step, rel_rec, rel_send,
                                                                             this_rel_type, hidden_n2e, hidden_e2n, flag)
                # last_pred, hidden_n2e, hidden_e2n = self.single_step_forward(this_step, rel_rec.double(),
                #                                                              rel_send.double(),
                #                                                              this_rel_type, hidden_n2e, hidden_e2n,
                #                                                              flag)
            preds.append(last_pred)

        sizes = [preds[0].size()[0], inputs.size()[1], preds[0].size()[2], preds[0].size()[3]]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()


class GRUDecoder_2(nn.Module):
    """
    GRU decoder module. Most recent implementation
    Based on the paper of Neural Relational Inference with Efficient Message Passing Mechanism
    """

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False):
        super(GRUDecoder_2, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first
        self.mlp1 = MLPfGRU(n_in=msg_out,
                            n_hid=msg_out,
                            n_out=msg_out,
                            do_prob=do_prob)
        self.mlp2 = MLPfGRU(n_in=msg_out + n_in_node,
                            n_hid=msg_out + n_in_node,
                            n_out=msg_out + n_in_node,
                            do_prob=do_prob)
        self.gru_edge = nn.GRU(msg_out, msg_out, batch_first=True)
        self.gru_node = nn.GRU(msg_out + n_in_node, msg_out + n_in_node, batch_first=True)

        self.hidden_er = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_ei = nn.Linear(n_hid, n_hid, bias=False)
        self.hidden_eh = nn.Linear(n_hid, n_hid, bias=False)

        self.input_er = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_ei = nn.Linear(n_in_node, n_hid, bias=True)
        self.input_en = nn.Linear(n_in_node, n_hid, bias=True)

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_nodes, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_nodes*(num_nodes-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([senders, receivers], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size()[0], pre_msg.size()[1],
                                        pre_msg.size()[2], self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # node to edge spatio-temporal message passing
        h_edges_1 = [self.mlp1(all_msgs[:, 0, i, :]).unsqueeze(0) for i in range(all_msgs.size()[2])]
        h_edges_1 = torch.cat(h_edges_1, dim=0)  # (49, 100, 256)
        h_edges_n = torch.empty(h_edges_1.size()[0], h_edges_1.size()[1], all_msgs.size()[1] - 1, h_edges_1.size()[-1])
        for i in range(h_edges_1.size()[0]):
            e1, hn = self.gru_edge(all_msgs[:, 1:, i, :], h_edges_1[i].unsqueeze(0))
            h_edges_n[i] = e1
        h_edges_n = torch.cat((h_edges_1.unsqueeze(0).view(h_edges_1.size()[1], 1,
                                                           h_edges_1.size()[0], h_edges_1.size()[-1]),
                               h_edges_n.view(h_edges_n.size()[1], h_edges_n.size()[2],
                                              h_edges_n.size()[0], h_edges_n.size()[-1])),
                              dim=1)

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # edge to node spatio-temporal message passing mechanism
        h_nodes_1 = [self.mlp2(aug_inputs[:, 0, i, :]).unsqueeze(0) for i in range(aug_inputs.size()[2])]
        h_nodes_1 = torch.cat(h_nodes_1, dim=0)  # (7, 100, 257)
        h_nodes_n = torch.empty(h_nodes_1.size()[0], h_nodes_1.size()[1], aug_inputs.size()[1] - 1,
                                h_nodes_1.size()[-1])
        for i in range(h_nodes_1.size()[0]):
            e1, hn = self.gru_node(aug_inputs[:, 1:, i, :], h_nodes_1[i].unsqueeze(0))
            h_nodes_n[i] = e1
        h_nodes_n = torch.cat((h_nodes_1.unsqueeze(0).view(h_nodes_1.size()[1], 1,
                                                           h_nodes_1.size()[0], h_nodes_1.size()[-1]),
                               h_nodes_n.view(h_nodes_n.size()[1], h_nodes_n.size()[2],
                                              h_nodes_n.size()[0], h_nodes_n.size()[-1])),
                              dim=1)
        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(h_nodes_n)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [rel_type.size()[0], inputs.size()[1], rel_type.size()[1],
                 rel_type.size()[2]]
        rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                 curr_rel_type)
            preds.append(last_pred)
            # print("last_pred:")
            # print(last_pred.size())
            # break

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()


class ATDecoder(nn.Module):
    """
    Attention based decoder module.
    Based on the paper of Neural Relational Inference with Efficient Message Passing Mechanism
    """

    def __init__(self, n_in_node, edge_types, n_hid, do_prob=0., skip_first=False):
        """

        :param n_in_node: the dimension of the features on a node
        :param edge_types: the edge types of the graph0
        :param n_hid: the number of hidden layers in MLP (and others)
        :param do_prob: the drop out probability
        :param skip_first: whether to skip the first sample point (in the test phase it was disabled)
        """
        super(ATDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, n_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)])
        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        self.mlp_AT_1 = MLP(n_in=n_hid, n_hid=n_hid, n_out=n_hid, do_prob=do_prob)
        self.mlp_AT_2 = MLP(n_in=n_hid, n_hid=n_hid, n_out=n_hid, do_prob=do_prob)
        self.mlp_AT_3 = MLP(n_in=n_hid, n_hid=n_hid, n_out=n_hid, do_prob=do_prob)
        self.mlp_AT_4 = MLP(n_in=n_hid, n_hid=n_hid, n_out=n_hid, do_prob=do_prob)
        self.mlp_AT_5 = MLP(n_in=n_hid + n_in_node, n_hid=n_hid + n_in_node, n_out=n_hid + n_in_node, do_prob=do_prob)
        self.mlp_AT_6 = MLP(n_in=n_hid + n_in_node, n_hid=n_hid + n_in_node, n_out=n_hid + n_in_node, do_prob=do_prob)
        self.mlp_AT_7 = MLP(n_in=n_hid + n_in_node, n_hid=n_hid + n_in_node, n_out=n_hid + n_in_node, do_prob=do_prob)
        self.mlp_AT_8 = MLP(n_in=n_hid + n_in_node, n_hid=n_hid + n_in_node, n_out=n_hid + n_in_node, do_prob=do_prob)

        self.out_fc1 = nn.Linear(n_in_node + n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        self.n_hid = n_hid
        self.n_in_node = n_in_node

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def bertha_cal(self, msgs, hidden_state, steps, met_size=0, edge_flag=True):
        """
        calculate the bertha coefficient for "Attention based implementation"
        :param met_size: the input size of the first data for compare, if default (=0), skip the comparison
        :param msgs: the current step input information (aggregated edges or nodes)
        :param hidden_state: the hidden states of the past n steps,
               with dimension: [steps, batch_size, steps in single partition, num_nodes/num_edges, num_features]
        :param steps: the current step number
        :param edge_flag: whether to calculate the coefficient for edge part or node part.
        :return: an array with [steps] x bertha value
        """
        msg_store = torch.zeros_like(hidden_state)
        num_len = msgs.size()[1]
        # print("num_len: {}".format(num_len))
        # print("hidden: {}".format(hidden_state.size()))
        if edge_flag:
            msg_m = [self.mlp_AT_3(msgs[:, ii, :, :]).unsqueeze(1) for ii in range(msgs.size()[1])]
            # print("msg_m: {}".format(len(msg_m)))
            # print(msg_m[0].size())
            msg_m = torch.cat(msg_m, dim=1)
            # print("after: msg_m: {}".format(msg_m.size()))
            d = msg_m.size()[0] * msg_m.size()[1] * msg_m.size()[2] * msg_m.size()[3]
            if met_size and met_size < num_len:
                msg_h = [hidden_state[j, :, :num_len, :, :] for j in range(steps)]
            else:
                msg_h = [hidden_state[j, :, :, :, :] for j in range(steps)]
            for ii, states in enumerate(msg_h):
                hid_e = [self.mlp_AT_4(states[:, k, :, :]).unsqueeze(1) for k in range(states.size()[1])]
                hid_e = torch.cat(hid_e, dim=1)
                msg_store[ii] = hid_e

        else:
            msg_m = [self.mlp_AT_7(msgs[:, ii, :, :]).unsqueeze(1) for ii in range(msgs.size()[1])]
            # print("msg_m: {}".format(len(msg_m)))
            # print(msg_m[0].size())
            msg_m = torch.cat(msg_m, dim=1)
            # print("after: msg_m: {}".format(msg_m.size()))
            d = msg_m.size()[0] * msg_m.size()[1] * msg_m.size()[2] * msg_m.size()[3]
            # msg_h = [mlp_AT_8(hidden_state[j, :, :, :, :]) for j in range(steps)]
            if met_size and met_size < num_len:
                msg_h = [hidden_state[j, :, :num_len, :, :] for j in range(steps)]
            else:
                msg_h = [hidden_state[j, :, :, :, :] for j in range(steps)]
            for ii, states in enumerate(msg_h):
                hid_e = [self.mlp_AT_8(states[:, k, :, :]).unsqueeze(1) for k in range(states.size()[1])]
                hid_e = torch.cat(hid_e, dim=1)
                msg_store[ii] = hid_e
        # print("msg_h: {}".format(len(msg_h)))
        # print(msg_h[0].size())
        msg_alpha = [torch.mul(msg_m, x) for x in msg_h]
        # print("msg_alpha length: {}".format(len(msg_alpha)))
        # print("msg_alpha[0]: {}".format(msg_alpha[0].size()))
        alpha = [torch.sum(element) / math.sqrt(d) for element in msg_alpha]
        berthas = [math.exp(jj) for jj in alpha]
        bertha_sum = torch.zeros(1)
        for entity in berthas:
            bertha_sum += entity
        berthas = [jjj / bertha_sum for jjj in berthas]
        berthas = [jjj.unsqueeze(0) for jjj in berthas]
        berthas = torch.cat(berthas, dim=0)
        return berthas

    def single_step_forward_ini(self, single_timestep_inputs, rel_rec, rel_send,
                                single_timestep_rel_type):
        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_nodes, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_nodes*(num_nodes-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)  # torch.Size([100, 5, 49, 1])
        senders = torch.matmul(rel_send, single_timestep_inputs)  # torch.Size([100, 5, 49, 1])
        pre_msg = torch.cat([senders, receivers], dim=-1)  # torch.Size([100, 5, 49, 2])

        # receivers = torch.matmul(rel_rec.double(), single_timestep_inputs.double())  # torch.Size([100, 5, 49, 1])
        # senders = torch.matmul(rel_send.double(), single_timestep_inputs.double())  # torch.Size([100, 5, 49, 1])
        # pre_msg = torch.cat([senders, receivers], dim=-1)  # torch.Size([100, 5, 49, 2])

        all_msgs = Variable(torch.zeros(pre_msg.size()[0], pre_msg.size()[1],
                                        pre_msg.size()[2], self.msg_out_shape))  # torch.Size([100, 5, 49, 256])
        # print(single_timestep_rel_type.size())  # torch.Size([100, 5, 49, 2])
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()
        start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))  # pre_msg.double()
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg
        # print("all_msgs:")  # torch.Size([100, 5, 49, 256])  # (num_batch, num_timesteps, num_edges, num_features)
        # print(all_msgs.size())  # torch.Size([100, 5, 49, 256])

        # Aggregate all msgs to receiver
        h_edges_1 = [self.mlp_AT_1(all_msgs[:, i, :, :]).unsqueeze(1) for i in
                     range(all_msgs.size()[1])]  # size: (100, 49, 256)
        # print(h_edges_1[0].size())  # torch.Size([100, 1, 49, 256])
        # print(len(h_edges_1))  # 5
        h_edges_1 = torch.cat(h_edges_1, dim=1)
        # print("h_edges_1: {}".format(h_edges_1.size()))  #  torch.Size([100, 5, 49, 256])

        # agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        # agg_msgs = agg_msgs.contiguous()  # torch.Size([100, 5, 49, 256])
        agg_msgs = h_edges_1.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        # agg_msgs = h_edges_1.transpose(-2, -1).matmul(rel_rec.double()).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()  # torch.Size([100, 5, 7, 256])

        # Skip connection
        aug_inputs = torch.cat([agg_msgs, single_timestep_inputs], dim=-1)  # torch.Size([100, 5, 7, 257])
        # print("aug_inputs:")  # torch.Size([100, 5, 7, 257])
        # print(aug_inputs.size())
        h_nodes_1 = [self.mlp_AT_5(aug_inputs[:, i, :, :]).unsqueeze(1) for i in range(aug_inputs.size()[1])]
        h_nodes_1 = torch.cat(h_nodes_1, dim=1)
        # print("h_nodes_1:")  # torch.Size([100, 5, 7, 257])
        # print(h_nodes_1.size())  # torch.Size([100, 5, 7, 257])

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(h_nodes_1)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred, h_edges_1.unsqueeze(0), h_nodes_1.unsqueeze(0)

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type, hidden_n2e, hidden_e2n,
                            met_size=0,
                            last_step_flag=False):
        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_nodes, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_nodes*(num_nodes-1), num_edge_types]

        # hidden_n2e and hidden_e2n have shape
        # [steps, batch_size, num_timesteps, num_nodes/num_connections, num_dims]
        if last_step_flag:
            size = single_timestep_inputs.size()[1]
            hidden_n2e = hidden_n2e[:, :, :size, :, :]
            hidden_e2n = hidden_e2n[:, :, :size, :, :]
        steps = len(hidden_e2n)
        s_steps = single_timestep_rel_type.size()[1]
        assert steps >= 1, "Check the input dimension of this step!"
        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)  # torch.Size([100, 5, 49, 1])
        senders = torch.matmul(rel_send, single_timestep_inputs)  # torch.Size([100, 5, 49, 1])
        pre_msg = torch.cat([senders, receivers], dim=-1)  # torch.Size([100, 5, 49, 2])

        # receivers = torch.matmul(rel_rec.double(), single_timestep_inputs.double())  # torch.Size([100, 5, 49, 1])
        # senders = torch.matmul(rel_send.double(), single_timestep_inputs.double())  # torch.Size([100, 5, 49, 1])
        # pre_msg = torch.cat([senders, receivers], dim=-1)  # torch.Size([100, 5, 49, 2])

        all_msgs = Variable(torch.zeros(pre_msg.size()[0], pre_msg.size()[1],
                                        pre_msg.size()[2], self.msg_out_shape))

        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()
        start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))  # pre_msg.double()
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg
        # print("all_msgs:")  # torch.Size([100, 5, 49, 256])  # (num_batch, num_timesteps, num_edges, num_features)

        # Aggregate all msgs to receiver

        # print(single_timestep_inputs.size())  # torch.Size([100, 5, 7, 1])
        # print(agg_msgs.size())  # torch.Size([100, 5, 7, 256])
        # attention score calculation for edges aggregation
        berthas_edges = self.bertha_cal(all_msgs, hidden_n2e, steps, met_size, edge_flag=True)

        hidden_present_edge = torch.zeros_like(all_msgs)
        s_ = min(steps, s_steps)
        for ind in range(s_):
            if ind == s_ - 1:
                x_t = [self.mlp_AT_2(all_msgs[:, k, :, :]).unsqueeze(1) for k in range(all_msgs.size()[1])]
                x_t = torch.cat(x_t, dim=1)
            else:
                x_t = [self.mlp_AT_2(hidden_n2e[ind, :, k, :, :]).unsqueeze(1) for k in range(hidden_n2e.size()[2])]
                x_t = torch.cat(x_t, dim=1)
            hidden_present_edge += berthas_edges[ind] * x_t

        agg_msgs = hidden_present_edge.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()  # torch.Size([100, 5, 7, 256])

        # Skip connection
        aug_inputs = torch.cat([agg_msgs, single_timestep_inputs], dim=-1)  # torch.Size([100, 5, 7, 257])

        # attention score calculation for nodes
        bertha_node = self.bertha_cal(aug_inputs, hidden_e2n, steps, met_size, edge_flag=False)

        hidden_present_node = torch.zeros_like(aug_inputs)
        for ind in range(s_):
            if ind == s_ - 1:
                # print("aug_inputs: {}".format(aug_inputs.size()))  # torch.Size([100, 5, 7, 257])
                x_t_n = [self.mlp_AT_6(aug_inputs[:, kk, :, :]).unsqueeze(1) for kk in range(aug_inputs.size()[1])]
                x_t_n = torch.cat(x_t_n, dim=1)
            else:
                # print("hidden_e2n: {}".format(hidden_e2n.size()))
                x_t_n = [self.mlp_AT_6(hidden_e2n[ind, :, kk, :, :]).unsqueeze(1) for kk in range(hidden_e2n.size()[2])]
                x_t_n = torch.cat(x_t_n, dim=1)
            hidden_present_node += bertha_node[ind] * x_t_n

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden_present_node)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred, hidden_present_edge.unsqueeze(0), hidden_present_node.unsqueeze(0)

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()  # torch.Size([100, 49, 7, 1])

        sizes = [rel_type.size()[0], inputs.size()[1], rel_type.size()[1],
                 rel_type.size()[2]]
        # print(rel_type.size())  # torch.Size([100, 49, 2])
        rel_type = rel_type.unsqueeze(1).expand(sizes)
        # print(rel_type.size())  # torch.Size([100, 49, 49, 2])

        time_steps = inputs.size()[1]
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        first_step = inputs[:, 0::pred_steps, :, :]  # torch.Size([100, 5, 7, 1])
        met_size = first_step.size()[1]
        # first_step = first_step.double()
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).
        hidden_edge = Variable(torch.zeros(inputs.size()[0],
                                           first_step.size()[1],
                                           rel_type.size()[2],
                                           self.n_hid))  # torch.Size([100, 5, 49, 256])
        # print("hidden_edge: {}".format(hidden_edge.size()))
        hidden_node = Variable(torch.zeros(inputs.size()[0],
                                           first_step.size()[1],
                                           first_step.size()[2],
                                           self.n_hid + self.n_in_node))  # [100, 5, 7, 257]
        # print("hidden_node: {}".format(hidden_node.size()))
        # print("-" * 35)
        # Run n prediction steps
        for step in range(0, pred_steps):

            if step == 0:
                # print(step)
                last_pred, hidden_edge, hidden_node = self.single_step_forward_ini(first_step, rel_rec, rel_send,
                                                                                   curr_rel_type)
                # print("hidden_edge: {}".format(hidden_edge.size()))
                # print("hidden_node: {}".format(hidden_node.size()))

            else:
                # print(step)
                this_step = inputs[:, step::pred_steps, :, :]  # torch.Size([100, 5, 7, 1])
                this_rel_type = rel_type[:, step::pred_steps, :, :]
                flag = first_step.size()[1] > this_step.size()[1]
                last_pred, hidden_edge_int, hidden_node_int = self.single_step_forward(this_step,
                                                                                       rel_rec,
                                                                                       rel_send,
                                                                                       this_rel_type,
                                                                                       hidden_edge,
                                                                                       hidden_node,
                                                                                       met_size,
                                                                                       flag)
                # last_pred, hidden_edge_int, hidden_node_int = self.single_step_forward(this_step, rel_rec.double(),
                #                                                                        rel_send.double(),
                #                                                                        this_rel_type, hidden_edge,
                #                                                                        hidden_node, flag)
                # print("before cat:")
                # print("hidden_edge: {}".format(hidden_edge.size()))
                # print("hidden_node: {}".format(hidden_node.size()))
                if step == pred_steps - 1:
                    continue
                hidden_edge = torch.cat((hidden_edge, hidden_edge_int))
                hidden_node = torch.cat((hidden_node, hidden_node_int))
                # print("hidden_edge: {}".format(hidden_edge.size()))
                # print("after cat:")
                # print("hidden_node: {}".format(hidden_node.size()))
            preds.append(last_pred)

        sizes = [preds[0].size()[0], inputs.size()[1],
                 preds[0].size()[2], preds[0].size()[3]]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            # print("i : {}".format(i))
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()


class VAE(nn.Module):
    """
    A wrapper for all the implementation of different types of encoders / decoders
        note : if use continuous relaxation. the encoder should output node features
    """
    def __init__(self, encoder, decoder, tau, hard, store_edges, rnn_decoder_flag, num_nodes, cuda_f,
                 c_r=False, c_r_layer=None):  # tau=args.temp, hard=args.hard
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tau = tau
        self.hard = hard
        self.num_nodes = num_nodes
        self.cuda_f = cuda_f
        self.store_edges = store_edges
        self.rnn_decoder_flag = rnn_decoder_flag
        self.c_r_flag = c_r
        self.c_r_layer = c_r_layer
        # self._enc_mu = torch.nn.Linear(100, 8)
        # self._enc_log_sigma = torch.nn.Linear(100, 8)
        self.edges_rec_sub = 0
        self.edges = None

    def _sample_latent(self, enc_out):
        """
        processing the output of encoder to sample from latent space
        """
        edges = gumbel_softmax(enc_out, tau=self.tau, hard=self.hard)
        self.edges = edges
        prob = my_softmax(enc_out, -1)
        self.prob = prob
        if self.store_edges:
            # print("*****************Sampled")
            edges_rec_sub = inter_check_(enc_out, self.num_nodes, self.cuda_f)
            self.edges_rec_sub = edges_rec_sub
            # todo:  at outside: edges_rec = np.concatenate((edges_rec, edges_rec_sub)) done
        return edges

    def forward(self, data, rel_rec, rel_send, timesteps, prediction_steps, step_i=0, graph_collapse=False):

        out_enc = self.encoder(data, rel_rec, rel_send)  # h_enc: 100

        # We let p_{theta}(z|x) be a mulltivariat Gaussian whose distribution paratmeters are computed from epsilon with
        # a MLP(a fully-connected neural network with a single hidden layer)
        if not self.c_r_flag:
            features = self._sample_latent(out_enc)  # 1-dim vector: len=8
        else:
            out_enc = out_enc.view(self.num_nodes, -1)
            features = self.c_r_layer(out_enc, step_i, graph_collapse)
            features = features.view(data.size()[0], data.size()[1], -1)

        if self.rnn_decoder_flag:
            out_dec = self.decoder(data, features, rel_rec, rel_send, 100,
                                   burn_in=True, burn_in_steps=timesteps - prediction_steps)
        elif self.c_r_flag:
            out_dec = self.decoder(data)
        else:
            out_dec = self.decoder(data, features, rel_rec, rel_send, prediction_steps)
        # return output, self.prob, self.edges_rec_sub, out_enc
        return out_dec, out_enc


class GraphConvolution(nn.Module):
    """
    Simple GCN layer,
    implementation from https://github.com/tkipf/pygcn/blob/1600b5b748b3976413d1e307540ccc62605b4d6d/pygcn/layers.py#L9
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size()[1])
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.matmul(inputs, self.weight)
        # output = torch.sparse.mm(adj, support)  # need torch.sparse.matmul (LOL)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttentionLayer_2(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer_2, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inputs, adj):
        h = inputs.view(-1, inputs.size()[-1])
        print("h: {}".format(h.size()))
        print("W: {}".format(self.W.size()))
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        print("Wh: {}".format(Wh.size()))
        a_input = self._prepare_attentional_mechanism_input(Wh)
        print("a_input: {}".format(a_input.size()))
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2).float())
        print("e: {}".format(e.size()))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention.float(), Wh.float())

        if self.concat:
            return F.elu(h_prime).view(inputs.size())
        else:
            return h_prime.view(inputs.size())

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Implementation from https://github.com/Diego999/pyGAT/blob/3664f2dc90cbf971564c0bf186dc794f12446d0c/layers.py#L7
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inputs, adj):
        # h = inputs.view(inputs.size()[0], -1)
        h = torch.squeeze(inputs)
        Wh = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime).view(inputs.size())
        else:
            return h_prime.view(inputs.size())

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


if __name__ == "__main__":
    print("This is modules.py")