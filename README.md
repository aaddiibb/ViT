# Vision Transformer (ViT) for MNIST Classification

## 📚 Table of Contents
1. [Introduction](#introduction)
2. [What is Vision Transformer?](#what-is-vision-transformer)
3. [Code Architecture Breakdown](#code-architecture-breakdown)
4. [How It Solves MNIST](#how-it-solves-mnist)
5. [Results](#results)

---

## Introduction
 This repository contains a step-by-step implementation of **Vision Transformer (ViT)** for classifying handwritten digits from the MNIST dataset. 

---
## Code Architecture Breakdown

### Step 1: Data Preparation
```python
# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transformation_operation)
test_dataset = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transformation_operation)

# Create data loaders for batching
train_data = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_data = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)
```

**What it does**: 
- Downloads MNIST dataset (28×28 grayscale images of digits 0-9)
- Converts images to tensors using transformations
- Groups data into batches of 64 images for efficient training

---

### Step 2: Patch Embedding (Converting Images to Patches)
```python
class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_channels=1,              # Grayscale image
            out_channels=128,           # Embedding dimension
            kernel_size=7,              # 7×7 patch size
            stride=7                    # Non-overlapping patches
        )
```

**What it does**:
- Takes a 28×28 image and divides it into 7×7 patches (4×4 = 16 patches total)
- Uses a convolutional layer to convert each patch into a 128-dimensional embedding
- **Result**: Each image becomes 16 patches, each represented as a 128-dimensional vector

**Why it works**: A 28×28 image with 7×7 patch size gives us (28÷7)² = 16 patches

---

### Step 3: Transformer Block (Self-Attention)
```python
class TransformerBlock(torch.nn.Module):
    def __init__(self, embedding_dim, attention_heads, mlp_dim, dropout_rate):
        super(TransformerBlock, self).__init__()
        # Multi-head self-attention
        self.attention = torch.nn.MultiheadAttention(embedding_dim, attention_heads, dropout=dropout_rate)
        
        # Feed-forward network
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, mlp_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(mlp_dim, embedding_dim)
        )
        
        # Layer normalization
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.norm2 = torch.nn.LayerNorm(embedding_dim)
```

**What it does**:

1. **Self-Attention**: Allows each patch to "look at" all other patches to understand relationships
   - Multi-head attention (4 heads) means the model learns 4 different types of relationships simultaneously
   - Example: One head might focus on edges, another on shapes, etc.

2. **Feed-Forward Network (MLP)**: Applies neural network transformation to each patch independently
   - Expands: 128 → 256 dimensions
   - Then compresses back: 256 → 128 dimensions
   - ReLU activation adds non-linearity

3. **Residual Connections + Layer Norm**: 
   - "Add": Connect input directly to output (residual connection)
   - "Norm": Normalize activations to keep training stable

**Flow**: Input → Attention → Add & Normalize → MLP → Add & Normalize → Output

---

### Step 4: MLPHead (Classification)
```python
class MLPHead(torch.nn.Module):
    def __init__(self, embedding_dim, mlp_nodes, num_classes, dropout_rate):
        super(MLPHead, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, mlp_nodes),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(mlp_nodes, 10)  # 10 classes (digits 0-9)
        )
    
    def forward(self, x):
        x = x[:, 0]  # Take [CLS] token
        x = self.mlp(x)
        return x
```

**What it does**:
- Takes the special [CLS] token (class token) which aggregates information from all patches
- Passes it through 2 fully-connected layers to predict which digit (0-9) is in the image

---

### Step 5: Complete ViT Model
```python
class ViT(torch.nn.Module):
    def __init__(self, embedding_dim, attention_heads, mlp_dim, dropout_rate, mlp_nodes, num_classes):
        super(ViT, self).__init__()
        
        # Convert image to patches
        self.patch_embed = PatchEmbedding()
        
        # Learnable [CLS] token (aggregates image information)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embedding_dim))
        
        # Learnable positional embeddings (tells model where patches are located)
        self.positional_embedding = torch.nn.Parameter(torch.zeros(1, patch_num + 1, embedding_dim))
        
        # Stack of 4 transformer blocks
        self.transformer_blocks = torch.nn.ModuleList(
            [TransformerBlock(embedding_dim, attention_heads, mlp_dim, dropout_rate) 
             for _ in range(4)]
        )
        
        # Classification head
        self.mlp_head = MLPHead(embedding_dim, mlp_nodes, num_classes, dropout_rate)
    
    def forward(self, x):
        # Step 1: Convert image to patch embeddings
        x = self.patch_embed(x)  # [batch, 16, 128]
        
        # Step 2: Add [CLS] token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch, 17, 128]
        
        # Step 3: Add positional embeddings
        x += self.positional_embedding
        
        # Step 4: Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Step 5: Classify using [CLS] token
        x = self.mlp_head(x)
        
        return x
```

**Complete Flow**:
1. Image (28×28) → Patches (16 patches of 128-dim each)
2. Add [CLS] token (17 items total)
3. Add positional information (so model knows patch locations)
4. Process through 4 transformer blocks (each patch attends to all others)
5. Use [CLS] token to predict digit (0-9)

---

## How It Solves MNIST

### The Problem
MNIST contains 70,000 images of handwritten digits (0-9), each 28×28 pixels. The task is to correctly identify which digit is in each image.

### The Solution Strategy

**1. Feature Extraction (Patches)**
- Instead of learning spatial hierarchies like CNNs, ViT immediately extracts all patch-level features
- 16 patches capture the entire digit structure

**2. Global Context (Self-Attention)**
- Each patch can attend to all other patches
- The model learns: "If I see a curved top and curved bottom patches together, it's probably an 8"
- This is more direct than stacking layers in CNNs

**3. Refined Learning (Transformer Blocks)**
- 4 transformer blocks refine the understanding progressively
- Each block can learn different aspects:
  - Block 1: Low-level shape patterns
  - Block 2: Part compositions
  - Block 3: Object-level features
  - Block 4: Final classification signals

**4. Decision Making (Classification Head)**
- The [CLS] token accumulates information from all patches
- The MLP head converts this to a probability distribution over 10 classes

### Training Process
```python
epochs = 10
for epoch in range(epochs):
    # For each batch of 64 images:
    for images, labels in train_data:
        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```



### Key Hyperparameters
```python
embedding_dim = 128      # Size of patch embeddings
attention_heads = 4      # Number of attention heads
mlp_dim = 256           # Hidden dimension in feed-forward
dropout_rate = 0.1      # Dropout for regularization
batch_size = 64         # Images per batch
epochs = 10             # Training iterations
```


### Sample Results
```
Epoch 1/10 - Train Loss: 0.35, Train Acc: 90.2%, Test Loss: 0.28, Test Acc: 91.5%
Epoch 2/10 - Train Loss: 0.18, Train Acc: 94.8%, Test Loss: 0.16, Test Acc: 94.9%
...
Epoch 10/10 - Train Loss: 0.08, Train Acc: 97.5%, Test Loss: 0.12, Test Acc: 97.8%
```

---


