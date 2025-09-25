import torch
import torch.nn.functional as F

# Example 1: Standard cross entropy (no event weights)
def standard_loss(out, targets):
    """Standard cross entropy loss"""
    return F.cross_entropy(out, targets)

# Example 2: Cross entropy with per-event weights
def weighted_cross_entropy(out, targets, event_weights):
    """
    Cross entropy loss with per-event weights
    
    Args:
        out: Model predictions (logits) [batch_size, n_classes]
        targets: True labels [batch_size]
        event_weights: Per-event weights [batch_size]
    
    Returns:
        Weighted loss (scalar)
    """
    # Calculate per-sample loss
    loss_per_sample = F.cross_entropy(out, targets, reduction='none')
    
    # Apply event weights
    weighted_loss = loss_per_sample * event_weights
    
    # Return mean of weighted losses
    return weighted_loss.mean()

# Example 3: In a training loop with PyTorch Geometric
def train_step_with_weights(model, batch, optimizer):
    """Training step with event weights from data.weight"""
    optimizer.zero_grad()
    
    # Forward pass
    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    
    # Check if weights are available
    if hasattr(batch, 'weight') and batch.weight is not None:
        # Use weighted loss
        loss = weighted_cross_entropy(out, batch.y, batch.weight)
    else:
        # Fall back to standard loss
        loss = F.cross_entropy(out, batch.y)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Example 4: Normalized weighted loss (ensures consistent scale)
def normalized_weighted_cross_entropy(out, targets, event_weights):
    """
    Normalized weighted cross entropy
    Normalizes weights so their sum equals batch size
    """
    # Normalize weights to sum to batch size
    normalized_weights = event_weights * len(event_weights) / event_weights.sum()
    
    # Calculate per-sample loss
    loss_per_sample = F.cross_entropy(out, targets, reduction='none')
    
    # Apply normalized weights
    weighted_loss = loss_per_sample * normalized_weights
    
    # Return mean
    return weighted_loss.mean()

# Example 5: Multi-class with class weights AND event weights
def combined_weighted_loss(out, targets, event_weights, class_weights=None):
    """
    Combines class weights and event weights
    
    Args:
        out: Model predictions [batch_size, n_classes]
        targets: True labels [batch_size]
        event_weights: Per-event weights [batch_size]
        class_weights: Per-class weights [n_classes]
    """
    if class_weights is not None:
        # Apply class weights first
        loss_per_sample = F.cross_entropy(out, targets, weight=class_weights, reduction='none')
    else:
        loss_per_sample = F.cross_entropy(out, targets, reduction='none')
    
    # Then apply event weights
    weighted_loss = loss_per_sample * event_weights
    
    return weighted_loss.mean()

# Example 6: Focal loss with event weights (for extreme imbalance)
def focal_loss_with_weights(out, targets, event_weights, alpha=0.25, gamma=2.0):
    """
    Focal loss with event weights
    Good for extreme class imbalance
    """
    # Get probabilities
    p = F.softmax(out, dim=1)
    
    # Get class probabilities
    ce_loss = F.cross_entropy(out, targets, reduction='none')
    p_t = p[torch.arange(len(targets)), targets]
    
    # Apply focal term
    focal_term = (1 - p_t) ** gamma
    focal_loss = focal_term * ce_loss
    
    # Apply event weights
    weighted_focal_loss = focal_loss * event_weights
    
    return weighted_focal_loss.mean()

# Example usage in actual training
if __name__ == "__main__":
    # Simulate data
    batch_size = 32
    n_classes = 4  # For multi-class
    
    # Model outputs (logits)
    out = torch.randn(batch_size, n_classes)
    
    # True labels
    targets = torch.randint(0, n_classes, (batch_size,))
    
    # Event weights (e.g., from cross-section, luminosity, etc.)
    event_weights = torch.rand(batch_size) + 0.5  # Random weights between 0.5 and 1.5
    
    # Calculate different losses
    loss_standard = standard_loss(out, targets)
    loss_weighted = weighted_cross_entropy(out, targets, event_weights)
    loss_normalized = normalized_weighted_cross_entropy(out, targets, event_weights)
    
    print(f"Standard loss: {loss_standard:.4f}")
    print(f"Weighted loss: {loss_weighted:.4f}")
    print(f"Normalized weighted loss: {loss_normalized:.4f}")
    
    # For binary classification
    out_binary = torch.randn(batch_size, 2)
    targets_binary = torch.randint(0, 2, (batch_size,))
    
    # Class weights for imbalanced classes (e.g., more background than signal)
    class_weights = torch.tensor([10.0, 1.0])  # Upweight class 0 (signal)
    
    loss_combined = combined_weighted_loss(out_binary, targets_binary, event_weights, class_weights)
    print(f"Combined weighted loss: {loss_combined:.4f}")
