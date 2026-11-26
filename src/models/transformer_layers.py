
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention adapted from ANIMATE's
    GraphMAE_transformer.py.

    Input:
        x: [batch_size, num_nodes, hidden_size]

    Output:
        [batch_size, num_nodes, hidden_size]
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        attention_dropout_rate=0.1
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads"
            )

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.scale = self.head_dim ** -0.5

        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)

        self.att_dropout = nn.Dropout(
            attention_dropout_rate
        )

        self.output_layer = nn.Linear(
            hidden_size,
            hidden_size
        )

    def forward(self, x, attn_bias=None):

        batch_size, num_nodes, hidden_size = x.shape

        # Generate Query, Key and Value
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)

        # Split into multiple attention heads
        q = q.view(
            batch_size, num_nodes,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size, num_nodes,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size, num_nodes,
            self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(
            q * self.scale,
            k.transpose(-2, -1)
        )

        # Optional graph structural attention bias
        if attn_bias is not None:
            scores = scores + attn_bias

        attention = torch.softmax(scores, dim=-1)

        attention = self.att_dropout(attention)

        output = torch.matmul(attention, v)

        # Merge attention heads
        output = output.transpose(1, 2).contiguous()

        output = output.view(
            batch_size,
            num_nodes,
            hidden_size
        )

        return self.output_layer(output)


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed Forward Network.
    """

    def __init__(self, hidden_size, ffn_size):
        super().__init__()

        self.layer1 = nn.Linear(
            hidden_size,
            ffn_size
        )

        self.activation = nn.GELU()

        self.layer2 = nn.Linear(
            ffn_size,
            hidden_size
        )

    def forward(self, x):

        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)

        return x


class EncoderLayer(nn.Module):
    """
    Transformer Encoder Layer.

    Contains:
    1. Multi-Head Self-Attention
    2. Residual Connection
    3. Layer Normalization
    4. Feed Forward Network
    """

    def __init__(
        self,
        hidden_size,
        ffn_size,
        num_heads,
        dropout_rate=0.1,
        attention_dropout_rate=0.1
    ):
        super().__init__()

        # Multi-Head Attention
        self.self_attention_norm = nn.LayerNorm(
            hidden_size
        )

        self.self_attention = MultiHeadAttention(
            hidden_size,
            num_heads,
            attention_dropout_rate
        )

        self.self_attention_dropout = nn.Dropout(
            dropout_rate
        )

        # Feed Forward Network
        self.ffn_norm = nn.LayerNorm(
            hidden_size
        )

        self.ffn = FeedForwardNetwork(
            hidden_size,
            ffn_size
        )

        self.ffn_dropout = nn.Dropout(
            dropout_rate
        )

    def forward(self, x, attn_bias=None):

        # 1. Multi-Head Self-Attention
        y = self.self_attention_norm(x)

        y = self.self_attention(
            y,
            attn_bias
        )

        y = self.self_attention_dropout(y)

        # 2. Residual Connection
        x = x + y

        # 3. Feed Forward Network
        y = self.ffn_norm(x)

        y = self.ffn(y)

        y = self.ffn_dropout(y)

        # 4. Residual Connection
        x = x + y

        return x
