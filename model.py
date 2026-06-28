import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class GraphConv(nn.Module):
    """
    Spatial Graph Convolutional Layer operating on multi-channel inputs.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bn = nn.BatchNorm1d(out_features)
        self.bias = Parameter(torch.FloatTensor(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, a):
        # x shape: [B, N, F_in], a shape: [B, N, N]
        support = torch.einsum('abc,abd->abd', a, x)
        outputs = torch.einsum('abd,de->abe', support, self.weight)
        if self.bias is not None:
            outputs = outputs + self.bias
        # Adjust for BatchNorm1d dimension format [B, C, N]
        outputs = self.bn(outputs.transpose(1, 2)).transpose(1, 2)
        return outputs

class GATAttention(nn.Module):
    def __init__(self, num_nodes, in_features, dropout=0.4, negative_slope=0.2):
        super(GATAttention, self).__init__()
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout)
        self.negative_slope = negative_slope


        self.att_weight = Parameter(torch.FloatTensor(in_features, 1))  # [F, 1]
        nn.init.xavier_uniform_(self.att_weight.data, gain=1.414)

    def forward(self, adj, features):
        # adj: [B, N, N]
        # features: [B, N, F]
        batch_size = adj.size(0)
        N = self.num_nodes


        feat_att = torch.matmul(features, self.att_weight)  # [B, N, 1]
        att_scores = feat_att + feat_att.transpose(1, 2)  # [B, N, N]
        att_scores = F.leaky_relu(att_scores, negative_slope=self.negative_slope)


        mask = (adj == 0).float() * (-1e9)
        att_scores = att_scores + mask  # [B, N, N]


        att_weights = F.softmax(att_scores, dim=-1)  # [B, N, N]
        att_weights = self.dropout(att_weights)


        adj_attended = adj * att_weights  # [B, N, N]
        return adj_attended, att_weights

class TemporalConv(nn.Module):
    """
    Temporal Convolutional Layer equipped with Gated Linear Units (GLU).
    """

    def __init__(self, in_channel, out_channel):
        super(TemporalConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)

    def forward(self, x):
        sig_re = torch.sigmoid(self.conv2(x))
        GLU_re = self.conv1(x).mul(sig_re)
        cov3_re = self.conv3(x)
        return GLU_re + cov3_re


class GCNBlock1(nn.Module):
    """
    Spatio-Temporal Graph Convolution Block for topological feature extraction.
    """

    def __init__(self, in_features, out_features, in_channel, out_channel, dropout_rate=0.4):
        super(GCNBlock1, self).__init__()
        self.tgc1 = TemporalConv(in_channel, out_channel)
        self.tgc2 = TemporalConv(out_channel, out_channel)
        self.sgc1 = GraphConv(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_channel)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, a_attended):
        x = self.tgc1(x)
        sgc1_re = F.relu(self.sgc1(x, a_attended))
        outputs = self.tgc2(sgc1_re)
        outputs = self.bn(outputs)
        return self.dropout(outputs)


class GCNBranch(nn.Module):
    """
    Graph Convolutional Branch utilizing High-Frequency Guided Topological Priors.
    """

    def __init__(self, input_size, GCN_hidden1=32, GCN_hidden3=10, dropout_rate=0.4):
        super(GCNBranch, self).__init__()
        _, _, self.num_nodes, self.seq_length = input_size

        self.input_proj = nn.Linear(self.seq_length, GCN_hidden1)
        self.input_proj1 = nn.Linear(GCN_hidden1, GCN_hidden3)

        self.gcn1 = GCNBlock1(self.seq_length, GCN_hidden1, self.num_nodes, self.num_nodes, dropout_rate)
        self.gcn3 = GCNBlock1(GCN_hidden1, GCN_hidden3, self.num_nodes, self.num_nodes, dropout_rate)

        self.feat_compress = nn.Sequential(
            nn.Linear(self.num_nodes * GCN_hidden3, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.intermediate_features = None

    def forward(self, x, a1, a2, a3, a4, a5, a6):
        x = x.squeeze(1)
        batch_size = x.size(0)

        # High-Frequency Guided Mask Generation: A_final = A_all * (Mask_KNN | Mask_PT)
        def apply_guidance(a_all, a_knn, a_pt):
            mask = torch.logical_or(a_knn > 0, a_pt > 0).float()
            return a_all * mask

        # Apply topological prior masks to each view
        a_plv_guided = apply_guidance(a1, a2, a3)
        a_pc_guided = apply_guidance(a4, a5, a6)

        # Multi-view topological matrix fusion
        a_fused = a_plv_guided  + a_pc_guided

        x1 = self.gcn1(x, a_fused)
        x1 = x1 + self.input_proj(x)
        x2 = self.gcn3(x1, a_fused)
        x2 = x2 + self.input_proj1(x1)

        self.intermediate_features = x2
        feat = self.feat_compress(x2.reshape(batch_size, -1))
        return feat


class CTAM(nn.Module):
    """
    Combined Temporal-Channel Attention Module for multi-scale feature enhancement.
    """

    def __init__(self, channel, reduction=4, temporal_kernel=15, kernel_size=3):
        super(CTAM, self).__init__()
        self.conv3x3 = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.relu_conv = nn.ReLU(inplace=True)
        self.conv_temporal = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=temporal_kernel, padding=(temporal_kernel - 1) // 2, bias=False),
            nn.Dropout1d(p=0.3)
        )
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=0.3),
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, bias=False)
        )
        self.ln_out = nn.LayerNorm(channel)
        self.avg_pool_global = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()
        self.sigmoid_final = nn.Sigmoid()

    def forward(self, x):
        b, c, t = x.shape
        x_conv = self.relu_conv(self.conv3x3(x))

        avg_pool_c = x.mean(dim=-1)
        max_pool_c = x.max(dim=-1)[0]
        channel_att_raw = (avg_pool_c + max_pool_c).reshape(b, c, 1)

        temporal_pool = torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)[0]], dim=1)
        temporal_att_raw = self.conv_temporal(temporal_pool)

        fusion_input = torch.cat([channel_att_raw.flatten(1), temporal_att_raw.flatten(1)], dim=1).unsqueeze(1)
        fusion_out = self.fusion_mlp(fusion_input).squeeze(1)

        channel_att = self.sigmoid(fusion_out[:, :c].reshape(b, c, 1))
        temporal_att = self.sigmoid(fusion_out[:, c:].reshape(b, 1, t))
        out = x * channel_att * temporal_att

        ln_out = self.ln_out(out.transpose(1, 2)).transpose(1, 2)
        pool_out = self.avg_pool_global(ln_out).permute(0, 2, 1)

        temporal_weight = self.sigmoid(torch.matmul(pool_out, x_conv))
        pool_x_conv = x_conv.mean(dim=1, keepdim=True).permute(0, 2, 1)
        channel_weight = self.sigmoid(torch.matmul(out, pool_x_conv))

        return out * self.sigmoid_final(channel_weight * temporal_weight)


class CNNBranch(nn.Module):
    """
    Temporal Convolutional Branch processing multi-scale downsampled sequences.
    """

    def __init__(self, n_chans, dropout_rate=0.3):
        super(CNNBranch, self).__init__()
        self.intermediate_features = None
        self.temp_conv1 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv2 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv3 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)
        self.temp_conv4 = nn.Conv1d(n_chans, n_chans, kernel_size=2, stride=2, groups=n_chans)

        self.chpool = nn.Sequential(
            nn.Conv1d(n_chans, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            nn.Conv1d(32, 32, kernel_size=4, groups=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
        )
        self.cbam_scale1 = CTAM(32, reduction=8, temporal_kernel=31)
        self.cbam_scale2 = CTAM(32, reduction=8, temporal_kernel=15)
        self.cbam_scale3 = CTAM(32, reduction=8, temporal_kernel=7)
        self.feat_fuse = nn.Sequential(
            nn.Linear(32 * 3, 96),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        temp_x = self.temp_conv1(x)
        temp_w1 = self.temp_conv2(temp_x)
        temp_w2 = self.temp_conv3(temp_w1)
        temp_w3 = self.temp_conv4(temp_w2)

        feat1 = self.cbam_scale1(self.chpool(temp_w1)).mean(dim=-1)
        feat2 = self.cbam_scale2(self.chpool(temp_w2)).mean(dim=-1)
        feat3 = self.cbam_scale3(self.chpool(temp_w3)).mean(dim=-1)

        self.intermediate_features = torch.cat([feat1, feat2, feat3], dim=1)
        return self.feat_fuse(self.intermediate_features)


class CombinedCNN_GCN(nn.Module):
    """
    HFG-Net: Dual-branch fusion model with adaptive sample-level gating network.
    """

    def __init__(self, num_classes=2, n_chans=21, input_size=(None, 1, 21, 1024), GCN_hidden1=32, GCN_hidden3=10,
                 dropout_rate=0.4):
        super(CombinedCNN_GCN, self).__init__()
        self.cnn_branch = CNNBranch(n_chans=n_chans, dropout_rate=dropout_rate)
        self.gcn_branch = GCNBranch(input_size=input_size, GCN_hidden1=GCN_hidden1, GCN_hidden3=GCN_hidden3,
                                    dropout_rate=dropout_rate)

        self.fusion_module = nn.Sequential(
            nn.Linear(96 * 2, 24),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(24, 2),
            nn.Softmax(dim=1)
        )
        self.final_classifier = nn.Sequential(
            nn.Linear(96, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.01),
            nn.Linear(32, num_classes)
        )
        self.cnn_norm = nn.LayerNorm(96)
        self.gcn_norm = nn.LayerNorm(96)
        self.fused_features = None

    def extract_cnn_features(self, cnn_x):
        """Extract multi-scale temporal features from CNN branch for visualization."""
        self.cnn_branch(cnn_x)
        return self.cnn_branch.intermediate_features

    def extract_gcn_features(self, gcn_x, a1, a2, a3, a4, a5, a6):
        """Extract high-frequency guided topological features from GCN branch for visualization."""
        self.gcn_branch(gcn_x, a1, a2, a3, a4, a5, a6)
        return self.gcn_branch.intermediate_features

    def forward(self, cnn_x, gcn_x, a1, a2, a3, a4, a5, a6, return_weights=False):
        cnn_feat = self.cnn_norm(self.cnn_branch(cnn_x))
        gcn_feat = self.gcn_norm(self.gcn_branch(gcn_x, a1, a2, a3, a4, a5, a6))

        # Dynamic sample-level feature integration via gating network
        weights = self.fusion_module(torch.cat([cnn_feat, gcn_feat], dim=1))
        fused_feat = cnn_feat * weights[:, 0:1] + gcn_feat * weights[:, 1:2]
        self.fused_features = fused_feat

        logits = self.final_classifier(fused_feat)

        # 增加条件返回逻辑
        if return_weights:
            return logits, weights
        return logits

    def extract_fused_features(self, cnn_x, gcn_x, a1, a2, a3, a4, a5, a6):
        """Extract integrated synergistic features for t-SNE manifold analysis."""
        self.forward(cnn_x, gcn_x, a1, a2, a3, a4, a5, a6)
        return self.fused_features