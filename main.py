import dgl
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
from dataset import Dataset
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score
from sklearn.model_selection import train_test_split
from cosine_annealing_warmup import CosineAnnealingWarmupLR
from multi_head_regularization_teacher_student import MultiHeadBWGNN_SA_RegularizedTeacherStudent
from train_regularized_contrastive import train_regularized_contrastive, analyze_regularized_head_specialization


def get_best_f1(labels, probs):

    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels.cpu())
        preds[probs[:,1].cpu() > thres] = 1
        mf1 = f1_score(labels.cpu(), preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


def calculate_all_f1_metrics(labels, preds):

    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
    f1_weighted = f1_score(labels, preds, average='weighted', zero_division=0)
    return f1_macro, f1_micro, f1_weighted


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Head Spectral-Adaptive BWGNN with Teacher-Student Regularization')
    parser.add_argument("--dataset", type=str, default="amazon",
                        help="Dataset: yelp/amazon/tfinance/tsocial/elliptic/tolokers")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order of polynomial filter")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs")
    parser.add_argument("--run", type=int, default=5, help="Number of runs")
    
    parser.add_argument("--contrastive_weight", type=float, default=0.01, help="Weight for contrastive loss (λ_contrast)")
    parser.add_argument("--contrastive_temp", type=float, default=0.1, help="Temperature for contrastive learning")
    parser.add_argument("--diversity_weight", type=float, default=0.05, help="Weight for diversity loss (λ_div)")
    parser.add_argument("--diversity_warmup_epochs", type=int, default=20, help="Epochs for diversity loss warm-up")
    parser.add_argument("--lr", type=float, default=0.015, help="Learning rate")
    parser.add_argument("--analyze_heads", action="store_true", help="Analyze head specialization after training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    
    parser.add_argument("--use_cosine_warmup", action="store_true", default=True, help="Use Cosine Annealing with Warm-up")
    parser.add_argument("--warmup_epochs", type=int, default=7, help="Warm-up epochs")
    parser.add_argument("--warmup_start_lr", type=float, default=1e-6, help="Starting LR for warm-up")
    parser.add_argument("--warmup_target_lr", type=float, default=None, help="Target LR after warm-up")
    parser.add_argument("--cosine_annealing_eta_min", type=float, default=5e-6, help="Min LR after cosine annealing")
    parser.add_argument("--warmup_type", type=str, default="linear", choices=["linear", "exponential", "polynomial"])

    args = parser.parse_args()
    args.model = 'multi_head_regularized_teacher_student'
    print(args)
    
    graph = Dataset(args.dataset, homo=True).graph
    in_feats = graph.ndata['feature'].shape[1]
    num_classes = 2

    if args.run == 1:
        print(f"--- Running Multi-Head Regularized Teacher-Student BWGNN (heads: {args.num_heads}) ---")
        print(f"    λ_contrast: {args.contrastive_weight}, λ_div: {args.diversity_weight}")
        print(f"    Temperature: {args.contrastive_temp}, Diversity warm-up: {args.diversity_warmup_epochs}")
        
        model = MultiHeadBWGNN_SA_RegularizedTeacherStudent(
            in_feats, args.hid_dim, num_classes, d=args.order, 
            num_heads=args.num_heads, contrastive_temp=args.contrastive_temp, 
            contrastive_weight=args.contrastive_weight, diversity_weight=args.diversity_weight,
            teacher_momentum=0.999
        )
        
        mf1, auc, loss_history = train_regularized_contrastive(model, graph, args)
        
        if args.analyze_heads:
            analyze_regularized_head_specialization(model, graph, graph.ndata['feature'], args)
    else:

        final_mf1s, final_aucs = [], []
        for tt in range(args.run):
            print(f"\n=== Run {tt+1}/{args.run} ===")
            model = MultiHeadBWGNN_SA_RegularizedTeacherStudent(
                in_feats, args.hid_dim, num_classes, d=args.order, 
                num_heads=args.num_heads, contrastive_temp=args.contrastive_temp, 
                contrastive_weight=args.contrastive_weight, diversity_weight=args.diversity_weight,
                teacher_momentum=0.999
            )
            mf1, auc, _ = train_regularized_contrastive(model, graph, args)
            final_mf1s.append(mf1)
            final_aucs.append(auc)
        
        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        
        print('\n=== Multi-Run Results Summary ===')
        print('F1-macro  - Mean: {:.2f}%, Std: {:.2f}%'.format(100 * np.mean(final_mf1s), 100 * np.std(final_mf1s)))
        print('AUC       - Mean: {:.2f}%, Std: {:.2f}%'.format(100 * np.mean(final_aucs), 100 * np.std(final_aucs)))

