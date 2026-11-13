"""Five-class VEPI IED spatial classifier with directed-flow characterization.

The encoder output Z has shape [B,T,C,D]. Temporal attention produces one
representation per channel, from which a low-rank Q-K projection constructs a
sample-dependent directed adjacency A. Here A[i,j] denotes source j -> target i.
The graph is summarized only by each channel's net directional strength,
out-strength minus in-strength. This single descriptor modulates the channel
representation before channel-attention readout.

The model returns logits [B,5] and adjacency [B,C,C]. The adjacency is a learned
asymmetric channel relation for classification rather than verified physiological
causality. Only the classification loss is required.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Left-padded convolution that never accesses future samples."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class CausalSeparableBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, dilation=1, dropout=0.1):
        super().__init__()
        self.stride = stride
        self.depthwise = CausalConv1d(
            channels,
            channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x[..., :: self.stride] if self.stride > 1 else x
        out = self.pointwise(self.depthwise(x))
        out = self.dropout(F.gelu(self.norm(out)))
        common = min(identity.size(-1), out.size(-1))
        return identity[..., :common] + out[..., :common]


class MultiScaleCausalEncoder(nn.Module):
    """Shared multi-scale encoder with explicit sharp-wave morphology cues."""

    def __init__(self, channels, node_dim=24, time_bins=16, dropout=0.15):
        super().__init__()
        if node_dim % 3 != 0:
            raise ValueError("node_dim必须能被3整除，以构建三尺度分支")
        self.channels = channels
        self.node_dim = node_dim
        self.time_bins = time_bins

        branch_dim = node_dim // 3
        self.stems = nn.ModuleList(
            CausalConv1d(1, branch_dim, kernel_size, stride=4)
            for kernel_size in (7, 15, 31)
        )
        self.stem_norm = nn.BatchNorm1d(node_dim)
        self.morphology_fusion = nn.Sequential(
            nn.Conv1d(node_dim * 2, node_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(node_dim),
            nn.GELU(),
        )
        # Preserve the slope morphology branch because its removal caused a
        # clear TUEV sensitivity/F1 decrease in the first-stage ablation.
        self.slope_stem = CausalConv1d(
            2, node_dim, kernel_size=7, stride=4, bias=False
        )
        self.slope_norm = nn.BatchNorm1d(node_dim)
        self.blocks = nn.Sequential(
            CausalSeparableBlock(
                node_dim, 9, stride=2, dilation=1, dropout=dropout
            ),
            CausalSeparableBlock(
                node_dim, 7, stride=2, dilation=2, dropout=dropout
            ),
        )

    def forward(self, x):
        batch, channels, length = x.shape
        if channels != self.channels:
            raise ValueError(
                f"模型期望 {self.channels} 通道，实际输入 {channels} 通道"
            )
        x = x.reshape(batch * channels, 1, length)

        raw_features = torch.cat([stem(x) for stem in self.stems], dim=1)
        raw_features = F.gelu(self.stem_norm(raw_features))

        first_difference = F.pad(x[..., 1:] - x[..., :-1], (1, 0))
        slope_input = torch.cat(
            [first_difference, first_difference.abs()],
            dim=1,
        )

        slope_features = F.gelu(
            self.slope_norm(self.slope_stem(slope_input))
        )

        x = self.morphology_fusion(
            torch.cat([raw_features, slope_features], dim=1)
        )

        x = self.blocks(x)

        # Keep the actual encoded length; time_bins is a legacy config field.
        encoded_length = x.size(-1)
        return x.view(
            batch, channels, self.node_dim, encoded_length
        ).permute(0, 3, 1, 2)


class DirectedTopologyReadout(nn.Module):
    """Characterize each channel by out-strength minus in-strength."""

    def __init__(self, node_dim, hidden_dim=48, rank_dim=12, channels=19):
        super().__init__()
        if min(node_dim, hidden_dim, rank_dim, channels) < 1:
            raise ValueError("All model dimensions must be positive")
        self.channels = channels
        self.rank_dim = rank_dim
        self.node_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.channel_embedding = nn.Parameter(
            torch.empty(1, 1, channels, hidden_dim)
        )
        nn.init.trunc_normal_(self.channel_embedding, std=0.02)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.time_score = nn.Linear(hidden_dim, 1)
        self.query = nn.Linear(hidden_dim, rank_dim, bias=False)
        self.key = nn.Linear(hidden_dim, rank_dim, bias=False)
        self.topology_proj = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.node_score = nn.Linear(hidden_dim + 1, 1)
        self.topology_scale_logit = nn.Parameter(
            torch.tensor(math.log(0.1 / 0.9))
        )
        self.register_buffer(
            "diagonal", torch.eye(channels, dtype=torch.bool), persistent=False
        )

    def _pool_time(self, state):
        """Soft salient pooling stabilised by a temporal mean."""
        weights = self.time_score(state).float().softmax(dim=1).to(state.dtype)
        return 0.5 * ((weights * state).sum(dim=1) + state.mean(dim=1))

    def _adjacency(self, nodes):
        """Return unnormalised A[target, source] in [0,1], with zero diagonal."""
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            query = self.query(nodes.float())
            key = self.key(nodes.float())
            scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.rank_dim)
            adjacency = scores.sigmoid()
            return adjacency.masked_fill(self.diagonal, 0.0)

    def _topology_descriptor(self, adjacency):
        """Return net directional strength as [B,C,1]."""
        denominator = max(self.channels - 1, 1)
        in_strength = adjacency.sum(dim=-1) / denominator
        out_strength = adjacency.sum(dim=-2) / denominator
        return (out_strength - in_strength).unsqueeze(-1)

    def forward(self, tokens):
        if (tokens.ndim != 4 or tokens.size(1) < 1
                or tokens.size(2) != self.channels):
            raise ValueError("tokens must be [B,T,C,D], T>=1, with configured C")
        state = self.node_proj(tokens) + self.channel_embedding
        nodes = self._pool_time(state)
        adjacency = self._adjacency(nodes)
        descriptor = self._topology_descriptor(adjacency)
        topology_update = self.topology_proj(descriptor.to(nodes.dtype))
        nodes = self.output_norm(
            nodes + self.topology_scale_logit.sigmoid() * topology_update
        )

        node_input = torch.cat([nodes, descriptor.to(nodes.dtype)], dim=-1)
        node_weights = self.node_score(node_input).float().softmax(dim=1)
        graph_embedding = (nodes * node_weights.to(nodes.dtype)).sum(dim=1)
        return graph_embedding, adjacency


class CausalIEDGraphNet(nn.Module):
    """IED classifier with a single directed-flow graph readout."""

    def __init__(self, cfg):
        super().__init__()
        self.encoder = MultiScaleCausalEncoder(
            cfg.channels, cfg.node_dim, getattr(cfg, "time_bins", None), cfg.dropout
        )
        self.graph = DirectedTopologyReadout(
            node_dim=cfg.node_dim,
            hidden_dim=cfg.graph_hidden,
            rank_dim=cfg.rank_dim,
            channels=cfg.channels,
        )
        self.classifier = nn.Sequential(
            nn.Linear(cfg.graph_hidden, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(64, 5),
        )

    def forward(self, x):
        tokens = self.encoder(x)
        graph_embedding, adjacency = self.graph(tokens)
        return self.classifier(graph_embedding), adjacency


def _smoke_test():
    """CPU forward/backward and graph semantics checks; no accuracy claims."""
    from types import SimpleNamespace

    torch.set_num_threads(2)
    torch.manual_seed(1)
    cfg = SimpleNamespace(channels=19, node_dim=24, time_bins=16,
                          graph_hidden=48, rank_dim=12, dropout=0.15)
    model = CausalIEDGraphNet(cfg)
    x = torch.randn(2, 19, 2000)
    logits, adjacency = model(x)
    assert logits.shape == (2, 5)
    assert adjacency.shape == (2, 19, 19)
    assert torch.isfinite(logits).all() and torch.isfinite(adjacency).all()
    assert torch.count_nonzero(adjacency.diagonal(dim1=-2, dim2=-1)) == 0
    assert torch.all((adjacency >= 0) & (adjacency <= 1))
    F.cross_entropy(logits, torch.tensor([0, 4])).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"Disconnected parameter: {name}"
        assert torch.isfinite(parameter.grad).all(), f"Nonfinite gradient: {name}"
    assert model.graph.query.weight.grad.abs().sum() > 0
    assert model.graph.key.weight.grad.abs().sum() > 0

    model.eval()
    with torch.no_grad():
        assert model.encoder(x).shape == (2, 125, 19, 24)
        assert model.encoder(x[..., :257]).shape == (2, 17, 19, 24)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            amp_logits, amp_adjacency = model(x[..., :257])
        assert torch.isfinite(amp_logits).all()
        assert torch.isfinite(amp_adjacency).all()
        assert amp_adjacency.dtype == torch.float32

    graph = DirectedTopologyReadout(24, channels=1)
    pooled, adjacency = graph(torch.randn(2, 3, 1, 24))
    assert torch.isfinite(pooled).all() and torch.count_nonzero(adjacency) == 0

    # Edge orientation: A[i,j] aggregates source j into target i.
    a = torch.tensor([[[[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]]])
    h = torch.tensor([[[[2.], [5.], [11.]]]])
    assert torch.equal((a @ h).flatten(), torch.tensor([5., 11., 2.]))

    # Verify net directional strength for A[target, source].
    graph = DirectedTopologyReadout(24, channels=3).eval()
    adjacency = torch.tensor(
        [[[0.0, 0.8, 0.2], [0.1, 0.0, 0.0], [0.4, 0.7, 0.0]]]
    )
    descriptor = graph._topology_descriptor(adjacency)
    expected = torch.tensor([[[-0.25], [0.70], [-0.45]]])
    assert torch.allclose(descriptor, expected, atol=1e-6)
    count = sum(p.numel() for p in model.parameters())
    print(f"Forward/backward/AMP and directed-flow checks OK, parameters={count:,}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        _smoke_test()
    else:
        parser.print_help()
