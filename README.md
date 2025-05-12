# Combining Large Language Model Embeddings and Graph Neural Networks for Accurate Job Recommendations

## Experiments

Each `.py` file in this repository represents a separate experiment. To run an experiment, place the required data files in the same directory as the script and execute:

```bash
python <experiment_name>.py
```

For example:

```bash
python Hybrid.py
```

Replace `Hybrid.py` with the name of the desired experiment script.

## Data

The dataset is stored in a private Google Drive folder and contains the author's proprietary data. Access to the folder(pre‑processed data and ready‑to‑use LLM embeddings) will be granted once the paper is accepted to preserve anonymity. Please follow the preprocessing steps detailed in the paper to generate the same data.

## Installation

Install required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure data files are in the script directory.
2. Activate your Python environment (optional but recommended).
3. Run your chosen experiment, e.g.:

   ```bash
   python Hybrid.py
   ```

## Embedding Generation

Embeddings are generated using the `gte-base-en-v1.5` model from Hugging Face’s Sentence-Transformers library. Below is a simplified example without multi-GPU processing, saving embeddings with NumPy:

```python
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Load the model and move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('gte-base-en-v1.5')
model.to(device)

# Example input texts (e.g., combined job descriptions)
batch_texts = jobs['CombinedText']  # pandas Series or list of strings

# Compute embeddings
torch_embeddings = model.encode(batch_texts, device=device, show_progress_bar=True)

# Convert to NumPy array and save
embeddings_np = np.array(torch_embeddings)
output_file_path = 'data/embeddings_jobs.npy'
np.save(output_file_path, embeddings_np)

print(f"Embeddings saved to: {output_file_path}")
```
