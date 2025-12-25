# Multi-Head Spectral-Adaptive GNN 

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Amazon dataset
python main.py --dataset amazon --train_ratio 0.01 --epoch 100  --num_heads 3  --analyze_heads  --run 10 --contrastive_weight 0.01  --diversity_weight 0.005 --lr 0.015  --cosine_annealing_eta_min 5e-6

# T-finance dataset
python main.py --dataset tfinance --train_ratio 0.01 --epoch 100 --num_heads 3 -analyze_heads  --run 10 --contrastive_weight 0.01 --diversity_weight 0.005 --lr 0.015 --cosine_annealing_eta_min 5e-6

# Elliptic dataset with custom parameters
python main.py --dataset elliptic --train_ratio 0.4 --epoch 100  --num_heads 3 --analyze_heads  --run 10 --contrastive_weight 0.01  --diversity_weight 0.005 --lr 0.015  --cosine_annealing_eta_min 5e-6   

## Key Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | amazon | amazon/tfinance/elliptic/tolokers |
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
- **Amazon/Tolokers**: DGL built-in fraud detection datasets
- **TFinance/Elliptic**: Place in `./dataset/` folder


