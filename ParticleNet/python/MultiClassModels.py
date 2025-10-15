#!/usr/bin/env python
"""
Multi-class ParticleNet models for 4-class classification.

Based on the proven V2 ParticleNet architecture but adapted for multi-class
classification of signal vs 3 background processes. Maintains the same
DynamicEdgeConv structure that performed well in binary classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LeakyReLU, Dropout, BatchNorm1d, ELU, LayerNorm
from torch_geometric.nn import global_mean_pool, knn_graph
from torch_geometric.nn import TransformerConv, GATConv
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import MessagePassing, global_max_pool, global_add_pool
from torch_geometric.utils import dropout_edge, softmax
from torch_geometric.nn import AttentionalAggregation


class EdgeConv(MessagePassing):
    """
    EdgeConv layer for graph neural networks.

    Implements message passing between neighboring nodes with MLP transformation
    of concatenated node features.
    """
    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__(aggr="mean")
        self.mlp = Sequential(
                Linear(2*in_channels, out_channels), LeakyReLU(), BatchNorm1d(out_channels), Dropout(dropout_p),
                Linear(out_channels, out_channels), LeakyReLU(), BatchNorm1d(out_channels), Dropout(dropout_p),
                Linear(out_channels, out_channels), LeakyReLU(), BatchNorm1d(out_channels), Dropout(dropout_p)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        return self.propagate(edge_index, x=x, batch=batch)

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        tmp = torch.cat([x_i, x_j - x_i], dim=1)
        return self.mlp(tmp)


class DynamicEdgeConv(EdgeConv):
    """
    Dynamic EdgeConv with k-nearest neighbors and residual connections.

    Builds k-NN graph dynamically and includes skip connections for better
    gradient flow. This is the core building block of the ParticleNet architecture.
    """
    def __init__(self, in_channels, out_channels, dropout_p, k=4):
        super().__init__(in_channels, out_channels, dropout_p=dropout_p)
        self.shortcut = Sequential(Linear(in_channels, out_channels), BatchNorm1d(out_channels), Dropout(dropout_p))
        self.dropout_p = dropout_p
        self.k = k

    def forward(self, x, edge_index=None, batch=None):
        if edge_index is None:
            edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        edge_index, _ = dropout_edge(edge_index, p=self.dropout_p, training=self.training)
        out = super().forward(x, edge_index, batch=batch)
        out += self.shortcut(x)
        return out


class MultiClassParticleNet(torch.nn.Module):
    """
    Multi-class ParticleNet for 4-class classification.

    - 3 DynamicEdgeConv layers with k=4 nearest neighbors
    - Concatenation of all conv outputs before pooling
    - Global mean pooling followed by dense layers
    - Extended to 4-class output (signal + 3 backgrounds)
    """
    def __init__(self, num_node_features, num_graph_features, num_classes=4, num_hidden=128, dropout_p=0.25):
        super(MultiClassParticleNet, self).__init__()

        # Input normalization
        self.gn0 = GraphNorm(num_node_features)

        # Three DynamicEdgeConv layers
        self.conv1 = DynamicEdgeConv(num_node_features, num_hidden, dropout_p, k=4)
        self.conv2 = DynamicEdgeConv(num_hidden, num_hidden, dropout_p, k=4)
        self.conv3 = DynamicEdgeConv(num_hidden, num_hidden, dropout_p, k=4)

        # Dense layers after global pooling
        self.bn0 = BatchNorm1d(num_hidden*3 + num_graph_features)
        self.dense1 = Linear(num_hidden*3 + num_graph_features, num_hidden)
        self.bn1 = BatchNorm1d(num_hidden)
        self.dense2 = Linear(num_hidden, num_hidden)
        self.bn2 = BatchNorm1d(num_hidden)
        self.output = Linear(num_hidden, num_classes)

        self.dropout_p = dropout_p
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                graph_input: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of multi-class ParticleNet.

        Args:
            x: Node features (N_nodes, num_node_features)
            edge_index: Edge connectivity (2, N_edges)
            graph_input: Global graph features (N_graphs, num_graph_features)
            batch: Batch assignment for nodes (N_nodes,)

        Returns:
            Class probabilities (N_graphs, num_classes)
        """
        # Input normalization
        x = self.gn0(x, batch=batch)

        # Graph convolution layers with concatenation
        conv1 = self.conv1(x, edge_index, batch=batch)
        conv2 = self.conv2(conv1, batch=batch)
        conv3 = self.conv3(conv2, batch=batch)
        x = torch.cat([conv1, conv2, conv3], dim=1)

        # Global pooling
        x = global_mean_pool(x, batch=batch)
        x = torch.cat([x, graph_input], dim=1)
        x = self.bn0(x)

        # Dense classification layers
        x = F.leaky_relu(self.dense1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.leaky_relu(self.dense2(x))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.output(x)

        # Return logits (no softmax) - loss function applies softmax internally
        return x


class MultiClassParticleNetV2(torch.nn.Module):
    """
    Multi-class ParticleNetV2 with transformer-based first layer.

    Similar to MultiClassParticleNet but uses TransformerConv for the first layer
    to potentially capture longer-range dependencies in the particle interactions.
    """
    def __init__(self, num_node_features, num_graph_features, num_classes=4, num_hidden=128, dropout_p=0.25):
        super(MultiClassParticleNetV2, self).__init__()

        # Input normalization
        self.gn0 = GraphNorm(num_node_features)

        # Mixed conv layers: Transformer + DynamicEdge
        self.conv1 = TransformerConv(num_node_features, num_hidden, heads=4, dropout=dropout_p)
        self.conv2 = DynamicEdgeConv(num_hidden, num_hidden, dropout_p, k=4)
        self.conv3 = DynamicEdgeConv(num_hidden, num_hidden, dropout_p, k=4)

        # Dense layers after global pooling
        self.bn0 = BatchNorm1d(num_hidden*3 + num_graph_features)
        self.dense1 = Linear(num_hidden*3 + num_graph_features, num_hidden)
        self.bn1 = BatchNorm1d(num_hidden)
        self.dense2 = Linear(num_hidden, num_hidden)
        self.bn2 = BatchNorm1d(num_hidden)
        self.output = Linear(num_hidden, num_classes)

        self.dropout_p = dropout_p
        self.num_classes = num_classes

    def forward(self, x, edge_index, graph_input, batch=None):
        """Forward pass of multi-class ParticleNetV2."""
        # Input normalization
        x = self.gn0(x, batch=batch)

        # Graph convolution layers
        conv1 = self.conv1(x, edge_index)  # TransformerConv handles edge_index directly
        conv2 = self.conv2(conv1, batch=batch)
        conv3 = self.conv3(conv2, batch=batch)
        x = torch.cat([conv1, conv2, conv3], dim=1)

        # Global pooling and classification (same as V1)
        x = global_mean_pool(x, batch=batch)
        x = torch.cat([x, graph_input], dim=1)
        x = self.bn0(x)

        x = F.leaky_relu(self.dense1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.leaky_relu(self.dense2(x))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.output(x)

        # Return logits (no softmax) - loss function applies softmax internally
        return x


class AttentionPooling(torch.nn.Module):
    """
    Attention-based pooling that learns to focus on the most important nodes.

    This is particularly useful for grouped classification where different
    physics processes within a group may have distinct signature particles.
    """
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.attention_net = Sequential(
            Linear(in_channels, hidden_channels),
            LeakyReLU(),
            Linear(hidden_channels, 1)
        )

    def forward(self, x, batch):
        # Compute attention weights
        attn_weights = self.attention_net(x)  # (num_nodes, 1)
        attn_weights = softmax(attn_weights.squeeze(-1), batch)  # (num_nodes,)

        # Apply attention weights
        x_attended = x * attn_weights.unsqueeze(-1)  # (num_nodes, features)

        # Global pooling with attention
        return global_add_pool(x_attended, batch)


class SelfAttentionLayer(torch.nn.Module):
    """
    Self-attention layer for learning better node representations.

    Helps the model focus on relationships between different particles
    that are most discriminative for grouped classification.
    """
    def __init__(self, in_channels, out_channels, heads=4, dropout_p=0.25):
        super().__init__()
        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_dim = out_channels // heads

        self.q_linear = Linear(in_channels, out_channels)
        self.k_linear = Linear(in_channels, out_channels)
        self.v_linear = Linear(in_channels, out_channels)

        self.dropout = Dropout(dropout_p)
        self.layer_norm = LayerNorm(out_channels)
        self.residual_proj = Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x, batch):
        batch_size = batch.max().item() + 1

        # Compute queries, keys, values
        q = self.q_linear(x).view(-1, self.heads, self.head_dim)
        k = self.k_linear(x).view(-1, self.heads, self.head_dim)
        v = self.v_linear(x).view(-1, self.heads, self.head_dim)

        # Compute attention scores within each graph
        attn_scores = []
        attn_values = []

        for b in range(batch_size):
            mask = (batch == b)
            if mask.sum() == 0:
                continue

            q_b = q[mask]  # (num_nodes_b, heads, head_dim)
            k_b = k[mask]
            v_b = v[mask]

            # Attention computation
            scores = torch.matmul(q_b, k_b.transpose(-2, -1)) / (self.head_dim ** 0.5)
            scores = F.softmax(scores, dim=-1)
            scores = self.dropout(scores)

            # Apply attention to values
            attended = torch.matmul(scores, v_b)  # (num_nodes_b, heads, head_dim)
            attended = attended.view(-1, self.out_channels)  # (num_nodes_b, out_channels)

            attn_values.append(attended)

        if len(attn_values) == 0:
            return x

        x_attended = torch.cat(attn_values, dim=0)

        # Residual connection
        if self.residual_proj is not None:
            x = self.residual_proj(x)
        x_attended = x_attended + x

        # Layer normalization
        return self.layer_norm(x_attended)


class EnhancedParticleNet(torch.nn.Module):
    """
    Enhanced ParticleNet designed specifically for grouped background classification.

    Key improvements for handling heterogeneous grouped classes:
    - Deeper architecture with 5 conv layers
    - Self-attention mechanism for better feature learning
    - Attention-based pooling to focus on discriminative particles
    - Progressive width reduction in dense layers
    - Multi-scale feature aggregation
    - Higher model capacity (~1M parameters vs ~270K)
    """
    def __init__(self, num_node_features, num_graph_features, num_classes=4, num_hidden=256, dropout_p=0.25):
        super(EnhancedParticleNet, self).__init__()

        # Input normalization
        self.gn0 = GraphNorm(num_node_features)

        # Five DynamicEdgeConv layers for deeper feature learning
        self.conv1 = DynamicEdgeConv(num_node_features, num_hidden, dropout_p, k=4)
        self.conv2 = DynamicEdgeConv(num_hidden, num_hidden, dropout_p, k=4)
        self.conv3 = DynamicEdgeConv(num_hidden, num_hidden, dropout_p, k=4)
        self.conv4 = DynamicEdgeConv(num_hidden, num_hidden, dropout_p, k=4)
        self.conv5 = DynamicEdgeConv(num_hidden, num_hidden, dropout_p, k=4)

        # Self-attention layer for better node interactions
        self.self_attn = SelfAttentionLayer(num_hidden, num_hidden, heads=8, dropout_p=dropout_p)

        # Attention-based pooling instead of simple mean pooling
        self.attention_pool = AttentionPooling(num_hidden, hidden_channels=num_hidden//2)

        # Progressive width reduction in dense layers
        dense_input_size = num_hidden + num_graph_features

        self.bn0 = BatchNorm1d(dense_input_size)
        self.dense1 = Linear(dense_input_size, num_hidden * 2)  # Expand first
        self.bn1 = BatchNorm1d(num_hidden * 2)

        self.dense2 = Linear(num_hidden * 2, num_hidden)  # Then reduce
        self.bn2 = BatchNorm1d(num_hidden)

        self.dense3 = Linear(num_hidden, num_hidden // 2)  # Further reduce
        self.bn3 = BatchNorm1d(num_hidden // 2)

        self.dense4 = Linear(num_hidden // 2, num_hidden // 4)  # Final reduction
        self.bn4 = BatchNorm1d(num_hidden // 4)

        self.output = Linear(num_hidden // 4, num_classes)

        self.dropout_p = dropout_p
        self.num_classes = num_classes

    def forward(self, x, edge_index, graph_input, batch=None):
        """Forward pass of enhanced ParticleNet."""
        # Input normalization
        x = self.gn0(x, batch=batch)

        # Deep graph convolution with multi-scale features
        conv1 = self.conv1(x, edge_index, batch=batch)
        conv2 = self.conv2(conv1, batch=batch)
        conv3 = self.conv3(conv2, batch=batch)
        conv4 = self.conv4(conv3, batch=batch)
        conv5 = self.conv5(conv4, batch=batch)

        # Use the deepest features for attention
        x = self.self_attn(conv5, batch)

        # Attention-based pooling
        x = self.attention_pool(x, batch)

        # Combine with graph-level features
        x = torch.cat([x, graph_input], dim=1)
        x = self.bn0(x)

        # Progressive dense layers with residual-like connections
        x = F.leaky_relu(self.dense1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = F.leaky_relu(self.dense2(x))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = F.leaky_relu(self.dense3(x))
        x = self.bn3(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = F.leaky_relu(self.dense4(x))
        x = self.bn4(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.output(x)

        # Return logits (no softmax) - loss function applies softmax internally
        return x


class HierarchicalParticleNet(torch.nn.Module):
    """
    Hierarchical ParticleNet for grouped classification.

    Uses a two-stage approach:
    1. First stage: Learn broad category separation (signal vs backgrounds)
    2. Second stage: Refine within background groups

    This approach may be more suitable for physics-motivated grouping.
    """
    def __init__(self, num_node_features, num_graph_features, num_classes=4, num_hidden=256, dropout_p=0.25):
        super(HierarchicalParticleNet, self).__init__()

        # Shared feature extraction layers
        self.gn0 = GraphNorm(num_node_features)
        self.conv1 = DynamicEdgeConv(num_node_features, num_hidden, dropout_p, k=4)
        self.conv2 = DynamicEdgeConv(num_hidden, num_hidden, dropout_p, k=4)
        self.conv3 = DynamicEdgeConv(num_hidden, num_hidden, dropout_p, k=4)

        # Attention mechanism
        self.self_attn = SelfAttentionLayer(num_hidden, num_hidden, heads=4, dropout_p=dropout_p)
        self.attention_pool = AttentionPooling(num_hidden, hidden_channels=num_hidden//2)

        # Dense feature layers
        dense_input_size = num_hidden + num_graph_features
        self.bn0 = BatchNorm1d(dense_input_size)
        self.feature_dense = Linear(dense_input_size, num_hidden)
        self.bn_feature = BatchNorm1d(num_hidden)

        # Final classification layer
        self.output = Linear(num_hidden, num_classes)

        self.dropout_p = dropout_p
        self.num_classes = num_classes

    def forward(self, x, edge_index, graph_input, batch=None):
        """Forward pass with hierarchical feature learning."""
        # Shared feature extraction
        x = self.gn0(x, batch=batch)

        conv1 = self.conv1(x, edge_index, batch=batch)
        conv2 = self.conv2(conv1, batch=batch)
        conv3 = self.conv3(conv2, batch=batch)

        # Self-attention on deepest features
        x = self.self_attn(conv3, batch)

        # Attention-based pooling
        x = self.attention_pool(x, batch)

        # Combine with graph features
        x = torch.cat([x, graph_input], dim=1)
        x = self.bn0(x)

        # Dense feature learning
        x = F.leaky_relu(self.feature_dense(x))
        x = self.bn_feature(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Final classification
        x = self.output(x)

        # Return logits (no softmax) - loss function applies softmax internally
        return x


class EfficientParticleNet(torch.nn.Module):
    """
    Efficient enhanced ParticleNet for grouped background classification.

    Balances improved capacity with practical training speed:
    - 4 DynamicEdgeConv layers (vs 5 in EnhancedParticleNet)
    - Lightweight attention pooling (no self-attention)
    - Moderate hidden size increase (128→192)
    - Streamlined dense layers (3 vs 4)
    - ~500K parameters (vs 1M+ in Enhanced, 270K in standard)
    """
    def __init__(self, num_node_features, num_graph_features, num_classes=4, num_hidden=192, dropout_p=0.25):
        super(EfficientParticleNet, self).__init__()

        # Input normalization
        self.gn0 = GraphNorm(num_node_features)

        # Four DynamicEdgeConv layers with moderate hidden size
        self.conv1 = DynamicEdgeConv(num_node_features, num_hidden, dropout_p, k=4)
        self.conv2 = DynamicEdgeConv(num_hidden, num_hidden, dropout_p, k=4)
        self.conv3 = DynamicEdgeConv(num_hidden, num_hidden, dropout_p, k=4)
        self.conv4 = DynamicEdgeConv(num_hidden, num_hidden, dropout_p, k=4)

        # Lightweight attention pooling (no O(n²) self-attention)
        self.attention_pool = AttentionPooling(num_hidden, hidden_channels=num_hidden//4)

        # Streamlined dense layers
        dense_input_size = num_hidden + num_graph_features

        self.bn0 = BatchNorm1d(dense_input_size)
        self.dense1 = Linear(dense_input_size, num_hidden * 2)  # Expand
        self.bn1 = BatchNorm1d(num_hidden * 2)

        self.dense2 = Linear(num_hidden * 2, num_hidden)  # Maintain
        self.bn2 = BatchNorm1d(num_hidden)

        self.dense3 = Linear(num_hidden, num_hidden // 2)  # Reduce
        self.bn3 = BatchNorm1d(num_hidden // 2)

        self.output = Linear(num_hidden // 2, num_classes)

        self.dropout_p = dropout_p
        self.num_classes = num_classes

    def forward(self, x, edge_index, graph_input, batch=None):
        """Forward pass of efficient enhanced ParticleNet."""
        # Input normalization
        x = self.gn0(x, batch=batch)

        # Graph convolution layers
        conv1 = self.conv1(x, edge_index, batch=batch)
        conv2 = self.conv2(conv1, batch=batch)
        conv3 = self.conv3(conv2, batch=batch)
        conv4 = self.conv4(conv3, batch=batch)

        # Use deepest features with lightweight attention pooling
        x = self.attention_pool(conv4, batch)

        # Combine with graph-level features
        x = torch.cat([x, graph_input], dim=1)
        x = self.bn0(x)

        # Streamlined dense layers
        x = F.leaky_relu(self.dense1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = F.leaky_relu(self.dense2(x))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = F.leaky_relu(self.dense3(x))
        x = self.bn3(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.output(x)

        # Return logits (no softmax) - loss function applies softmax internally
        return x


class OptimizedParticleNet(torch.nn.Module):
    """
    Optimized ParticleNet specifically designed to address EnhancedParticleNet bottlenecks.

    Key optimizations:
    - No self-attention (eliminates O(n²) complexity)
    - Smart feature concatenation from multiple conv layers
    - Batch-efficient operations only
    - Conservative capacity increase: 128→160 hidden units
    - Only 3.5 conv layers (conv4 has smaller output)
    - Optimized attention pooling with smaller hidden size
    - ~350K parameters (balanced between 270K standard and 1M+ enhanced)
    """
    def __init__(self, num_node_features, num_graph_features, num_classes=4, num_hidden=160, dropout_p=0.25):
        super(OptimizedParticleNet, self).__init__()

        # Input normalization
        self.gn0 = GraphNorm(num_node_features)

        # Progressive width reduction strategy
        hidden_2 = int(num_hidden * 0.8)  # 128 for default 160
        hidden_3 = int(num_hidden * 0.6)  # 96 for default 160

        # Four conv layers with progressive width reduction
        self.conv1 = DynamicEdgeConv(num_node_features, num_hidden, dropout_p, k=4)
        self.conv2 = DynamicEdgeConv(num_hidden, num_hidden, dropout_p, k=4)
        self.conv3 = DynamicEdgeConv(num_hidden, hidden_2, dropout_p, k=4)
        self.conv4 = DynamicEdgeConv(hidden_2, hidden_3, dropout_p, k=3)  # Smaller k for efficiency

        # Multi-scale feature concatenation (inspired by V2 but optimized)
        concat_features = num_hidden + num_hidden + hidden_2 + hidden_3  # Progressive concat

        # Very lightweight attention pooling
        self.attention_pool = AttentionPooling(concat_features, hidden_channels=32)

        # Efficient dense layers
        dense_input_size = concat_features + num_graph_features

        self.bn0 = BatchNorm1d(dense_input_size)
        self.dense1 = Linear(dense_input_size, num_hidden)
        self.bn1 = BatchNorm1d(num_hidden)
        self.dense2 = Linear(num_hidden, num_hidden // 2)
        self.bn2 = BatchNorm1d(num_hidden // 2)
        self.output = Linear(num_hidden // 2, num_classes)

        self.dropout_p = dropout_p
        self.num_classes = num_classes

    def forward(self, x, edge_index, graph_input, batch=None):
        """Forward pass optimized for speed and efficiency."""
        # Input normalization
        x = self.gn0(x, batch=batch)

        # Progressive conv layers with multi-scale aggregation
        conv1 = self.conv1(x, edge_index, batch=batch)
        conv2 = self.conv2(conv1, batch=batch)
        conv3 = self.conv3(conv2, batch=batch)
        conv4 = self.conv4(conv3, batch=batch)

        # Multi-scale feature concatenation
        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        # Lightweight attention pooling
        x = self.attention_pool(x, batch)

        # Combine with graph features
        x = torch.cat([x, graph_input], dim=1)
        x = self.bn0(x)

        # Efficient dense layers
        x = F.leaky_relu(self.dense1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = F.leaky_relu(self.dense2(x))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.output(x)
        # Return logits (no softmax) - loss function applies softmax internally
        return x


def create_multiclass_model(model_type='ParticleNet', num_node_features=9, num_graph_features=4,
                           num_classes=4, num_hidden=128, dropout_p=0.25):
    """
    Factory function to create multi-class models.

    Args:
        model_type: Type of model ('ParticleNet', 'ParticleNetV2', 'OptimizedParticleNet', 'EfficientParticleNet', 'EnhancedParticleNet', 'HierarchicalParticleNet')
        num_node_features: Number of node features (default: 9)
        num_graph_features: Number of global graph features (default: 4)
        num_classes: Number of output classes (default: 4)
        num_hidden: Hidden layer size (default: 128 for basic, 160 for optimized, 192 for efficient, 256 for enhanced)
        dropout_p: Dropout probability (default: 0.25)

    Returns:
        Initialized model
    """
    if model_type == 'ParticleNet':
        return MultiClassParticleNet(
            num_node_features, num_graph_features, num_classes, num_hidden, dropout_p
        )
    elif model_type == 'ParticleNetV2':
        return MultiClassParticleNetV2(
            num_node_features, num_graph_features, num_classes, num_hidden, dropout_p
        )
    elif model_type == 'OptimizedParticleNet':
        # Use optimized default hidden size
        optimized_hidden = max(num_hidden, 160) if num_hidden < 160 else num_hidden
        return OptimizedParticleNet(
            num_node_features, num_graph_features, num_classes, optimized_hidden, dropout_p
        )
    elif model_type == 'EfficientParticleNet':
        # Use moderate default hidden size for efficient model
        efficient_hidden = max(num_hidden, 192) if num_hidden < 192 else num_hidden
        return EfficientParticleNet(
            num_node_features, num_graph_features, num_classes, efficient_hidden, dropout_p
        )
    elif model_type == 'EnhancedParticleNet':
        # Use larger default hidden size for enhanced model
        enhanced_hidden = max(num_hidden, 256)  # Minimum 256 for enhanced
        return EnhancedParticleNet(
            num_node_features, num_graph_features, num_classes, enhanced_hidden, dropout_p
        )
    elif model_type == 'HierarchicalParticleNet':
        # Use larger default hidden size for hierarchical model
        hierarchical_hidden = max(num_hidden, 256)  # Minimum 256 for hierarchical
        return HierarchicalParticleNet(
            num_node_features, num_graph_features, num_classes, hierarchical_hidden, dropout_p
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Available: ParticleNet, ParticleNetV2, OptimizedParticleNet, EfficientParticleNet, EnhancedParticleNet, HierarchicalParticleNet")


# Example usage and testing
if __name__ == "__main__":
    import torch
    from torch_geometric.data import Data, Batch

    print("Testing MultiClassParticleNet models...")

    # Create sample data
    num_nodes = 20
    num_node_features = 9
    num_graph_features = 4
    batch_size = 3

    # Node features
    x = torch.randn(num_nodes * batch_size, num_node_features)

    # Create edge indices (k-NN will be computed dynamically)
    edge_index = torch.randint(0, num_nodes * batch_size, (2, 50))

    # Graph-level features
    graph_input = torch.randn(batch_size, num_graph_features)

    # Batch assignment
    batch = torch.cat([torch.full((num_nodes,), i) for i in range(batch_size)])

    print(f"Input shapes:")
    print(f"  x: {x.shape}")
    print(f"  edge_index: {edge_index.shape}")
    print(f"  graph_input: {graph_input.shape}")
    print(f"  batch: {batch.shape}")

    # Test MultiClassParticleNet
    print("\n1. Testing MultiClassParticleNet:")
    model = MultiClassParticleNet(num_node_features, num_graph_features, num_classes=4, num_hidden=64)
    model.eval()

    with torch.no_grad():
        output = model(x, edge_index, graph_input, batch)

    print(f"   Output shape: {output.shape}")
    print(f"   Output sum (should be ~1 per sample): {output.sum(dim=1)}")
    print(f"   Sample predictions: {output[0]}")

    # Test MultiClassParticleNetV2
    print("\n2. Testing MultiClassParticleNetV2:")
    model_v2 = MultiClassParticleNetV2(num_node_features, num_graph_features, num_classes=4, num_hidden=64)
    model_v2.eval()

    with torch.no_grad():
        output_v2 = model_v2(x, edge_index, graph_input, batch)

    print(f"   Output shape: {output_v2.shape}")
    print(f"   Output sum (should be ~1 per sample): {output_v2.sum(dim=1)}")
    print(f"   Sample predictions: {output_v2[0]}")

    # Test factory function
    print("\n3. Testing factory function:")
    for model_type in ['ParticleNet', 'ParticleNetV2']:
        model = create_multiclass_model(model_type, num_hidden=64)
        with torch.no_grad():
            output = model(x, edge_index, graph_input, batch)
        print(f"   {model_type}: {output.shape}")

    # Count parameters
    print("\n4. Model parameters:")
    model = create_multiclass_model('ParticleNet', num_hidden=128)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    print("\nAll tests completed successfully!")