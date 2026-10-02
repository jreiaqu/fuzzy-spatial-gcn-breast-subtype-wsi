"""
GCN model for WSI-level molecular subtype prediction (context-aware branch).

PatchGCN_MeanMax_LSelec: GENConv layers (DeepGCN residual blocks) over the
patch graph, followed by gated attention pooling and a linear classifier.
Only GENConv + attention pooling is implemented; `gnn_layer_type`, `pooling`
and `edge_agg` are kept in the signature for compatibility with the
reference scripts.

Edge weighting (`use_edge_features=True`):
  Option 1 (precomputed fuzzy graphs)
    fuzzy_combined       uses graph.edge_index_fuzzy / graph.edge_mu_fuzzy
  Option 2 (weights computed in forward on the inherited spatial k-NN graph)
    spatial              1 - d_s
    morphological        1 - d_m
    spatial_fuzzy        exp(-d_s^2 / 2 sigma_s^2)
    morphological_fuzzy  exp(-d_m^2 / 2 sigma_m^2)
    combined_fuzzy       alpha * mu_s + (1 - alpha) * mu_m
  with d_s = graph.edge_features and d_m = graph.edge_feat_dist.

`use_edge_features=False` (default) is the topology-only GENConv of the
reference work; it is needed to load weights/bcnb_*_ca_genconv_attn.pth,
which have no `lin_edge` parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GENConv, DeepGCNLayer
from torch.nn import ReLU, LayerNorm


class PatchGCN_MeanMax_LSelec(torch.nn.Module):
    def __init__(self, num_layers=4, edge_agg='spatial',
                 num_features=1024, hidden_dim=128, pooling="attention",
                 dropout=0.25, n_classes=4, gnn_layer_type='GENConv',
                 edge_mode='spatial', sigma_spatial=0.5, sigma_morphological=0.5, alpha=0.5,
                 use_edge_features=False):
        super(PatchGCN_MeanMax_LSelec, self).__init__()
        self.pooling = pooling
        self.edge_agg = edge_agg
        self.edge_mode = edge_mode
        self.sigma_spatial = sigma_spatial
        self.sigma_morphological = sigma_morphological
        self.alpha = alpha  # weight of mu_s in combined_fuzzy; (1 - alpha) for mu_m
        self.num_layers = num_layers - 1
        self.num_features = num_features
        self.use_edge_features = use_edge_features

        self.fc = nn.Sequential(*[nn.Linear(self.num_features, hidden_dim), nn.ReLU(), nn.Dropout(0.25)])

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer',
                           edge_dim=(1 if self.use_edge_features else None))
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res', dropout=0.1, ckpt_grad=i % 3)
            self.layers.append(layer)

        self.path_phi = nn.Sequential(*[nn.Linear(hidden_dim * num_layers, hidden_dim * num_layers), nn.ReLU(), nn.Dropout(0.25)])
        self.path_attention_head = Attn_Net_Gated(L=hidden_dim * num_layers, D=hidden_dim * num_layers, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(hidden_dim * num_layers, hidden_dim * num_layers), nn.ReLU(), nn.Dropout(dropout)])

        self.classifier = torch.nn.Linear(hidden_dim * num_layers, n_classes)

    def _edge_inputs(self, graph):
        """Return (edge_index, edge_attr) according to the edge weighting mode."""
        # getattr defaults keep models pickled before these attributes existed loadable
        if not getattr(self, 'use_edge_features', True):
            return graph['edge_index'], None

        edge_mode = getattr(self, 'edge_mode', 'spatial')
        sigma_s = getattr(self, 'sigma_spatial', 0.5)
        sigma_m = getattr(self, 'sigma_morphological', 0.5)
        alpha = getattr(self, 'alpha', 0.5)

        # Option 1: membership precomputed by generate_fuzzy_graphs.py
        if edge_mode == 'fuzzy_combined':
            return graph['edge_index_fuzzy'], graph['edge_mu_fuzzy'].float().unsqueeze(-1)

        # Option 2: weights computed from the distances stored on the spatial k-NN edges
        d_s = graph['edge_features']
        if d_s.dim() > 1:
            d_s = d_s.squeeze(-1)

        if edge_mode == 'spatial':
            w = 1.0 - d_s
        elif edge_mode == 'morphological':
            w = 1.0 - graph['edge_feat_dist']
        elif edge_mode == 'spatial_fuzzy':
            w = torch.exp(-d_s**2 / (2 * sigma_s**2))
        elif edge_mode == 'morphological_fuzzy':
            d_m = graph['edge_feat_dist']
            w = torch.exp(-d_m**2 / (2 * sigma_m**2))
        elif edge_mode == 'combined_fuzzy':
            d_m = graph['edge_feat_dist']
            mu_s = torch.exp(-d_s**2 / (2 * sigma_s**2))
            mu_m = torch.exp(-d_m**2 / (2 * sigma_m**2))
            w = alpha * mu_s + (1.0 - alpha) * mu_m
        else:
            raise ValueError(f"Unknown edge_mode: '{edge_mode}'")
        return graph['edge_index'], w.unsqueeze(-1)

    def forward(self, graph):
        edge_index, edge_attr = self._edge_inputs(graph)

        x = self.fc(graph['x'])
        x_ = x

        x = self.layers[0].conv(x_, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)
        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)

        h_path = x_
        h_path = self.path_phi(h_path)

        A_path, h_path = self.path_attention_head(h_path)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1), h_path)
        h = self.path_rho(h_path).squeeze()
        logits = self.classifier(h).unsqueeze(0)  # [1 x n_classes]
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        return Y_prob, Y_hat, logits, h


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


# ---------------------------------------------------------------------------
# NCA classes (VGG16 + attention MIL), taken from the earlier group code
# (MIL_models.py) so that weights/bcnb_*_nca_vgg16_attn.pth, which were
# pickled against the module name "MIL_models", can be loaded. Protocol 2
# (retrain_classifier_predictions.py) only uses .milAggregation and
# .classifier.
# ---------------------------------------------------------------------------

import torchvision


class Encoder(torch.nn.Module):

    def __init__(self, freeze_bb_weights=False, pretrained=True, backbone='resnet18', aggregation=False):
        super(Encoder, self).__init__()

        self.aggregation = aggregation
        self.pretrained = pretrained
        self.backbone = backbone
        self.freeze_bb_weights = freeze_bb_weights

        if backbone == 'resnet18':
            resnet = torchvision.models.resnet18(pretrained=self.pretrained)
            self.F = torch.nn.Sequential(resnet.conv1,
                                         resnet.bn1,
                                         resnet.relu,
                                         resnet.maxpool,
                                         resnet.layer1,
                                         resnet.layer2,
                                         resnet.layer3,
                                         resnet.layer4)
        if backbone == 'resnet50':
            resnet50 = torchvision.models.resnet50(pretrained=self.pretrained)
            self.F = resnet50

        elif backbone == 'vgg16':
            vgg16 = torchvision.models.vgg16(pretrained=self.pretrained)
            self.F = vgg16.features
        elif backbone == 'vgg16_bn':
            vgg16_bn = torchvision.models.vgg16_bn(pretrained=self.pretrained)
            self.F = vgg16_bn.features
        elif backbone == 'vgg19':
            vgg19 = torchvision.models.vgg19(pretrained=self.pretrained)
            self.F = vgg19.features

        if self.freeze_bb_weights:
            for param in self.F.parameters():
                param.requires_grad = False

        self.gradients = None

    def forward(self, x):
        out = self.F(x)

        if self.backbone == "vgg16" or self.backbone == "vgg19" or self.backbone == "vgg16_bn":
            out = torch.nn.AdaptiveAvgPool2d((1, 1))(out)

        return out

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.features_conv(x)

    def activations_hook(self, grad):
        self.gradients = grad


class MILAttention(torch.nn.Module):
    def __init__(self, input_dim):
        super(MILAttention, self).__init__()

        # Attention MIL embedding from Ilse et al. (2018) for MIL.
        self.L = input_dim
        self.D = 128
        self.K = 1
        self.attention_V = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Tanh()
        )
        self.attention_U = torch.nn.Sequential(
            torch.nn.Linear(self.L, self.D),
            torch.nn.Sigmoid()
        )

        self.attention_weights = torch.nn.Linear(self.D, self.K)

    def forward(self, features):
        A_V = self.attention_V(features)
        A_U = self.attention_U(features)
        w = torch.softmax(self.attention_weights(A_V * A_U), dim=0)
        features = torch.transpose(features, 1, 0)
        embedding = torch.squeeze(torch.mm(features, w))
        return embedding, w


class MILAggregation(torch.nn.Module):
    def __init__(self, backbone, aggregation='mean', nClasses=1, mode='embedding'):
        super(MILAggregation, self).__init__()

        self.mode = mode
        self.aggregation = aggregation
        self.nClasses = nClasses
        self.backbone = backbone

        if self.aggregation == 'attention':
            if self.backbone == 'vgg16':
                input_dim = 512
            elif self.backbone == 'resnet50':
                input_dim = 1000

            self.attention_pooling = MILAttention(input_dim=input_dim)

    def forward(self, feats):
        if self.aggregation == 'max':
            embedding = torch.max(feats, dim=0)[0]
            return embedding
        elif self.aggregation == 'attention':
            embedding, attention_weights = self.attention_pooling(feats)
            return embedding
        elif self.aggregation == 'mean':
            embedding = torch.mean(feats, dim=0)
            return embedding


class MILArchitecture(torch.nn.Module):

    def __init__(self, classes, freeze_bb_weights=False, pretrained=True, mode='embedding', aggregation='mean', backbone='vgg19', include_background=False):
        super(MILArchitecture, self).__init__()

        self.classes = classes
        self.n_classes = len(classes)
        self.mode = mode
        self.aggregation = aggregation
        self.backbone = backbone
        self.include_background = include_background
        self.C = []
        self.prototypical = False
        self.pretained = pretrained
        self.freeze_bb_weights = freeze_bb_weights

        if self.include_background:
            self.nClasses = len(classes) + 1
        else:
            self.nClasses = len(classes)
        self.eps = 1e-6

        self.bb = Encoder(pretrained=self.pretained, backbone=self.backbone, aggregation=True, freeze_bb_weights=self.freeze_bb_weights)

        if self.backbone == 'vgg16':
            self.classifier = torch.nn.Linear(512, self.nClasses)
        if self.backbone == 'vgg16_bn':
            self.classifier = torch.nn.Linear(512, self.nClasses)
        elif self.backbone == 'vgg19':
            self.classifier = torch.nn.Linear(512, self.nClasses)
        elif self.backbone == 'resnet50':
            self.classifier = torch.nn.Linear(1000, self.nClasses)

        self.milAggregation = MILAggregation(aggregation=aggregation, nClasses=self.nClasses, mode=self.mode, backbone=self.backbone)

    def forward(self, images):
        features = self.bb(images)

        if self.mode == 'instance':
            patch_classification = torch.softmax(self.classifier(torch.squeeze(features)), 1)
            global_classification = self.milAggregation(patch_classification)

        if self.mode == 'embedding' or self.mode == 'mixed':
            if features.shape[0] > 1:
                embedding = self.milAggregation(torch.squeeze(features))
            else:
                embedding = torch.squeeze(self.milAggregation(features))

            global_classification = self.classifier(torch.squeeze(embedding))
            patch_classification = self.classifier(torch.squeeze(features))

        if self.include_background:
            global_classification = global_classification[1:]

        logits = global_classification
        Y_hat = torch.argmax(logits)
        Y_prob = F.softmax(logits, dim=0)

        return Y_prob, Y_hat, logits
