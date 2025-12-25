# Multi-Head Spectral-Adaptive GNN with Teacher-Student Regularization

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Amazon dataset
python main.py --dataset amazon --num_heads 4 --epoch 100

# Elliptic dataset with custom parameters
python main.py --dataset elliptic --train_ratio 0.4 --epoch 100 \
    --num_heads 3 --contrastive_weight 0.01 --diversity_weight 0.005 \
    --lr 0.015 --analyze_heads --run 10

# Multiple runs for statistical significance
python main.py --dataset yelp --num_heads 4 --run 5
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | amazon | yelp/amazon/tfinance/tsocial/elliptic/tolokers |
| `--num_heads` | 4 | Number of spectral-adaptive heads |
| `--order` | 2 | Chebyshev polynomial order |
| `--epoch` | 100 | Training epochs |
| `--train_ratio` | 0.4 | Training data ratio |
| `--contrastive_weight` | 0.1 | λ_contrast for contrastive loss |
| `--diversity_weight` | 0.05 | λ_div for diversity loss |
| `--contrastive_temp` | 0.1 | Contrastive learning temperature |
| `--lr` | 0.01 | Learning rate |
| `--run` | 5 | Number of runs |
| `--analyze_heads` | false | Analyze head specialization |


## Datasets

- **Yelp/Amazon**: DGL built-in fraud detection datasets
- **TFinance/TSocial/Elliptic**: Place in `./dataset/` folder


