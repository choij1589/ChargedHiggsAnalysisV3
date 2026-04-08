# ParticleNet Architecture

Source: `python/lib/MultiClassModels.py`

## Data Flow

```
Node features (N, 9)  Graph features (G, 8)
        |
    GraphNorm
        |
  DynamicEdgeConv (k=4) → conv1 (N, 128)
        |
  DynamicEdgeConv (k=4) → conv2 (N, 128)
        |
  DynamicEdgeConv (k=4) → conv3 (N, 128)
        |
  cat[conv1, conv2, conv3] → (N, 384)
        |
  global mean pool → (G, 384)
        |
  cat[pooled, graph_features] → (G, 392) → BatchNorm
        |
  Linear → LeakyReLU → BN → Dropout  (G, 128)
        |
  Linear → LeakyReLU → BN → Dropout  (G, 128)
        |
  Linear → (G, 4) raw logits
          [signal, nonprompt, diboson, ttX]
```

## DynamicEdgeConv

```
x (N, C)
  ├─ build k-NN graph → edge dropout (p)
  │       |
  │  cat[x_i, x_j - x_i] per edge
  │       |
  │  3 × (Linear → LeakyReLU → BN → Dropout)
  │       |
  │  mean over neighbors
  │
  └─ shortcut: Linear → BN → Dropout
        |
       (+) → output (N, C')
```

## Hyperparameters

| Parameter | Default |
|-----------|---------|
| node features | 9 |
| graph features | 8 |
| num classes | 4 |
| hidden size | 128 |
| dropout | 0.25 |
| k (NN) | 4 |
