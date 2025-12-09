# Configuration Module

This module stores configuration constants and embeddings used across the application.

## Beauty Standard Embedding

The beauty standard embedding is a 768-dimensional vector representing the ideal beauty standard used for product ranking.

### Loading the Embedding

The embedding is loaded via `get_beauty_standard_embedding()` which checks in this order:

1. **Config metadata** (passed via LangGraph config)
2. **Environment variable** (`BEAUTY_STANDARD_EMBEDDING`)
3. **File path** (`BEAUTY_STANDARD_EMBEDDING_PATH`, defaults to `data/beauty_standard_embedding.npy`)
4. **Placeholder** (zero vector) if none of the above are found

### Setting the Embedding

To save a beauty standard embedding:

```python
from app.config.beauty_standard import set_beauty_standard_embedding
import numpy as np

# Your 768-dim embedding vector
embedding = np.array([...])  # 768 values

# Save to default path (data/beauty_standard_embedding.npy)
set_beauty_standard_embedding(embedding)

# Or save to custom path
set_beauty_standard_embedding(embedding, save_path="path/to/embedding.npy")
```

### Environment Variables

- `BEAUTY_STANDARD_EMBEDDING`: Base64 or comma-separated embedding values (TODO: implement parsing)
- `BEAUTY_STANDARD_EMBEDDING_PATH`: Path to `.npy` file containing the embedding

### File Format

The embedding should be saved as a NumPy array file (`.npy` format) with shape `(768,)`.


