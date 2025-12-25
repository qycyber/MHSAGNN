import torch
import torch.nn.functional as F
import numpy as np
import time
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split

def train_regularized_contrastive(model, g, args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    features = g.ndata['feature']
    labels = g.ndata['label']
    index = list(range(len(labels)))
    if args.dataset == 'amazon':
        index = list(range(3305, len(labels)))

    if args.dataset == 'elliptic':
        labeled_mask = (labels == 0) | (labels == 1)
        index = [i for i in index if labeled_mask[i]]

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=args.train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()

    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())

    initial_lr = args.lr if hasattr(args, 'lr') else 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    if getattr(args, 'use_cosine_warmup', True):
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from cosine_annealing_warmup import CosineAnnealingWarmupLR

        warmup_epochs = getattr(args, 'warmup_epochs', 7)
        warmup_start_lr = getattr(args, 'warmup_start_lr', 1e-6)
        warmup_target_lr = getattr(args, 'warmup_target_lr', None)
        if warmup_target_lr is None:
            warmup_target_lr = initial_lr
        cosine_annealing_eta_min = getattr(args, 'cosine_annealing_eta_min', 1e-6)
        warmup_type = getattr(args, 'warmup_type', 'linear')

        scheduler = CosineAnnealingWarmupLR(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=args.epoch,
            warmup_start_lr=warmup_start_lr,
            warmup_target_lr=warmup_target_lr,
            eta_min=cosine_annealing_eta_min,
            warmup_type=warmup_type,
            verbose=False
        )
        print(f'Using Cosine Annealing with Warm-up:')
        print(f'  Warm-up ({warmup_epochs} epochs): {warmup_start_lr:.2e} → {warmup_target_lr:.2e} ({warmup_type})')
        print(f'  Annealing ({args.epoch - warmup_epochs} epochs): {warmup_target_lr:.2e} → {cosine_annealing_eta_min:.2e}')
        use_metric_based_scheduler = False
    elif getattr(args, 'use_reduce_lr_on_plateau', False):
        mode = getattr(args, 'lr_scheduler_mode', 'max')
        factor = getattr(args, 'lr_scheduler_factor', 0.5)
        patience = getattr(args, 'lr_scheduler_patience', 10)
        threshold = getattr(args, 'lr_scheduler_threshold', 1e-4)
        min_lr = getattr(args, 'min_lr', None)
        if min_lr is None:
            min_lr = initial_lr * 0.001

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr,
            verbose=False
        )
        print(f'Using ReduceLROnPlateau (legacy): mode={mode}, factor={factor}, patience={patience}, min_lr={min_lr}')
        use_metric_based_scheduler = True
    elif getattr(args, 'use_cosine_annealing', False):
        min_lr = getattr(args, 'min_lr', None)
        if min_lr is None:
            min_lr = initial_lr * 0.01
        T_max = args.epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=min_lr, verbose=False
        )
        print(f'Using CosineAnnealingLR (legacy): lr from {initial_lr} to {min_lr} over {T_max} epochs')
        use_metric_based_scheduler = False
    else:
        scheduler = None
        print(f'Using fixed learning rate: {initial_lr}')
        use_metric_based_scheduler = False

    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.
    final_tmicro_f1, final_tweighted_f1 = 0., 0.

    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    class_weight = torch.tensor([1., weight])
    print('cross entropy weight: ', weight)
    print('contrastive weight λ_contrast: ', getattr(args, 'contrastive_weight', 0.1))
    print('diversity weight λ_div: ', getattr(args, 'diversity_weight', 0.05))

    loss_history = {
        'total': [],
        'classification': [],
        'contrastive': [],
        'diversity': []
    }

    diversity_warmup_epochs = getattr(args, 'diversity_warmup_epochs', 20)

    time_start = time.time()
    for e in range(args.epoch):
        model.train()

        if args.model in ['multi_head_regularized_teacher_student', 'multi_head_true_teacher']:
            logits, losses = model(g, features, return_losses=True)

            classification_loss = F.cross_entropy(
                logits[train_mask],
                labels[train_mask],
                weight=class_weight
            )

            contrastive_loss = losses.get('contrastive_loss', torch.tensor(0.0))
            diversity_loss = losses.get('diversity_loss', torch.tensor(0.0))

            if e < diversity_warmup_epochs:
                diversity_warmup_factor = e / diversity_warmup_epochs
            else:
                diversity_warmup_factor = 1.0

            total_loss = model.compute_total_loss(
                logits[train_mask],
                labels[train_mask],
                {
                    'contrastive_loss': contrastive_loss,
                    'diversity_loss': diversity_loss * diversity_warmup_factor
                },
                class_weight=class_weight
            )

            loss_history['total'].append(total_loss.item())
            loss_history['classification'].append(classification_loss.item())
            loss_history['contrastive'].append(contrastive_loss.item())
            loss_history['diversity'].append(diversity_loss.item())

        elif args.model in ['multi_head_contrastive']:
            logits, contrastive_loss = model(g, features, return_contrastive_loss=True)

            classification_loss = F.cross_entropy(
                logits[train_mask],
                labels[train_mask],
                weight=class_weight
            )

            total_loss = model.compute_total_loss(
                logits[train_mask],
                labels[train_mask],
                contrastive_loss,
                class_weight=class_weight
            )

            loss_history['total'].append(total_loss.item())
            loss_history['classification'].append(classification_loss.item())
            loss_history['contrastive'].append(contrastive_loss.item())
            loss_history['diversity'].append(0.0)

        elif args.model in ['multi_head_teacher_student']:
            logits, contrastive_loss = model(g, features, return_contrastive_loss=True)

            classification_loss = F.cross_entropy(
                logits[train_mask],
                labels[train_mask],
                weight=class_weight
            )

            total_loss = model.compute_total_loss(
                logits[train_mask],
                labels[train_mask],
                contrastive_loss,
                class_weight=class_weight
            )

            loss_history['total'].append(total_loss.item())
            loss_history['classification'].append(classification_loss.item())
            loss_history['contrastive'].append(contrastive_loss.item())
            loss_history['diversity'].append(0.0)

        elif args.model in ['bwgnn_sa', 'bwgnn_sa_sparse', 'bwgnn_sa_sparse_fixed',
                           'bwgnn_sa_sparse_batch', 'cheb_baseline', 'multi_head_sa', 'multi_head_channel']:
            logits = model(g, features)
            total_loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weight)
            classification_loss = total_loss
            contrastive_loss = torch.tensor(0.0)
            diversity_loss = torch.tensor(0.0)

            loss_history['total'].append(total_loss.item())
            loss_history['classification'].append(classification_loss.item())
            loss_history['contrastive'].append(0.0)
            loss_history['diversity'].append(0.0)

        else:
            logits = model(features)
            total_loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=class_weight)
            classification_loss = total_loss
            contrastive_loss = torch.tensor(0.0)
            diversity_loss = torch.tensor(0.0)

            loss_history['total'].append(total_loss.item())
            loss_history['classification'].append(classification_loss.item())
            loss_history['contrastive'].append(0.0)
            loss_history['diversity'].append(0.0)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if args.model in ['multi_head_regularized_teacher_student', 'multi_head_true_teacher']:
                eval_logits = model(g, features, return_losses=False)
            elif args.model in ['multi_head_contrastive']:
                eval_logits = model(g, features, return_contrastive_loss=False)
            elif args.model in ['multi_head_teacher_student']:
                eval_logits = model(g, features, return_contrastive_loss=False)
            elif args.model in ['bwgnn_sa', 'bwgnn_sa_sparse', 'bwgnn_sa_sparse_fixed',
                               'bwgnn_sa_sparse_batch', 'cheb_baseline', 'multi_head_sa', 'multi_head_channel']:
                eval_logits = model(g, features)
            else:
                eval_logits = model(features)

        probs = eval_logits.softmax(1)
        f1, thres = get_best_f1(labels[val_mask], probs[val_mask].detach())
        preds = np.zeros_like(labels)
        preds[probs[:, 1].detach() > thres] = 1
        trec = recall_score(labels[test_mask], preds[test_mask], zero_division=0)
        tpre = precision_score(labels[test_mask], preds[test_mask], zero_division=0)
        tmf1, tmicro_f1, tweighted_f1 = calculate_all_f1_metrics(labels[test_mask], preds[test_mask])
        tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())

        if scheduler is not None:
            if use_metric_based_scheduler:
                scheduler.step(f1)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = initial_lr

        if best_f1 < f1:
            best_f1 = f1
            final_trec = trec
            final_tpre = tpre
            final_tmf1 = tmf1
            final_tmicro_f1 = tmicro_f1
            final_tweighted_f1 = tweighted_f1
            final_tauc = tauc

        if e % 10 == 0:
            phase_info = ""
            if scheduler is not None and hasattr(scheduler, 'get_current_phase'):
                lr_phase = scheduler.get_current_phase()
                phase_progress = scheduler.get_phase_progress()
                phase_info = f" [{lr_phase.capitalize()} {phase_progress*100:.0f}%]"

            if args.model in ['multi_head_regularized_teacher_student', 'multi_head_true_teacher']:
                print('Epoch {}, total_loss: {:.4f} (cls: {:.4f}, contr: {:.4f}, div: {:.4f}), val mf1: {:.4f}, (best {:.4f}), lr: {:.6f}{}'.format(
                    e, total_loss.item(), classification_loss.item(), contrastive_loss.item(),
                    diversity_loss.item() * diversity_warmup_factor, f1, best_f1, current_lr, phase_info))
            elif args.model in ['multi_head_contrastive', 'multi_head_teacher_student']:
                print('Epoch {}, total_loss: {:.4f} (cls: {:.4f}, contr: {:.4f}), val mf1: {:.4f}, (best {:.4f}), lr: {:.6f}{}'.format(
                    e, total_loss.item(), classification_loss.item(), contrastive_loss.item(), f1, best_f1, current_lr, phase_info))
            else:
                print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f}), lr: {:.6f}{}'.format(e, total_loss.item(), f1, best_f1, current_lr, phase_info))

    time_end = time.time()
    print('time cost: ', time_end - time_start, 's')

    if args.model in ['multi_head_regularized_teacher_student', 'multi_head_true_teacher'] and len(loss_history['contrastive']) > 0:
        print('\n=== 正则化对比学习训练分析 ===')
        avg_cls_loss = np.mean(loss_history['classification'][-10:])
        avg_contr_loss = np.mean(loss_history['contrastive'][-10:])
        avg_div_loss = np.mean(loss_history['diversity'][-10:])
        total_reg_loss = avg_contr_loss + avg_div_loss

        print('最终平均分类损失: {:.4f}'.format(avg_cls_loss))
        print('最终平均对比损失: {:.4f}'.format(avg_contr_loss))
        print('最终平均多样性损失: {:.4f}'.format(avg_div_loss))
        print('对比损失占比: {:.2f}%'.format(100 * avg_contr_loss / (avg_cls_loss + total_reg_loss)))
        print('多样性损失占比: {:.2f}%'.format(100 * avg_div_loss / (avg_cls_loss + total_reg_loss)))
        print('总正则化损失占比: {:.2f}%'.format(100 * total_reg_loss / (avg_cls_loss + total_reg_loss)))

    elif args.model in ['multi_head_contrastive', 'multi_head_teacher_student'] and len(loss_history['contrastive']) > 0:
        print('\n=== 对比学习训练分析 ===')
        avg_cls_loss = np.mean(loss_history['classification'][-10:])
        avg_contr_loss = np.mean(loss_history['contrastive'][-10:])
        print('最终平均分类损失: {:.4f}'.format(avg_cls_loss))
        print('最终平均对比损失: {:.4f}'.format(avg_contr_loss))
        print('对比损失占比: {:.2f}%'.format(100 * avg_contr_loss / (avg_cls_loss + avg_contr_loss)))

    print('\n=== Test Results ===')
    print('  Recall: {:.2f}%'.format(final_trec*100))
    print('  Precision: {:.2f}%'.format(final_tpre*100))
    print('  F1-macro: {:.2f}%'.format(final_tmf1*100))
    print('  F1-micro: {:.2f}%'.format(final_tmicro_f1*100))
    print('  F1-weighted: {:.2f}%'.format(final_tweighted_f1*100))
    print('  AUC: {:.2f}%'.format(final_tauc*100))

    return final_tmf1, final_tauc, loss_history

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

def analyze_regularized_head_specialization(model, g, features, args):

    if args.model not in ['multi_head_regularized_teacher_student', 'multi_head_true_teacher', 'multi_head_contrastive', 'multi_head_teacher_student']:
        print("头部专门化分析仅适用于多头对比学习模型")
        return None

    print("\n=== 头部专门化分析 ===")
    model.eval()
    with torch.no_grad():
        if args.model in ['multi_head_regularized_teacher_student', 'multi_head_true_teacher']:
            analysis = model.get_detailed_regularization_analysis(g, features)

            print(f"头数: {analysis['num_heads']}")
            print(f"专门化分数: {analysis['head_specialization_score']:.3f} (越接近1.0越好)")
            print(f"对比学习温度: {analysis['diversity_analysis']['num_heads']}")

            print("\n=== 多样性正则化分析 ===")
            diversity_info = analysis['diversity_analysis']
            print(f"多样性分数: {diversity_info['diversity_score']:.3f} (越接近1.0越好)")
            print(f"对角线偏差: {diversity_info['diagonal_deviation']:.3f} (越接近0越好)")
            print(f"非对角线和: {diversity_info['off_diagonal_sum']:.3f} (越接近0越好)")

            print("\n各头频段专门化情况:")
            for head_name, head_info in analysis['head_theta_analysis'].items():
                print(f"  {head_name}: {head_info['frequency_focus']}")
                print(f"    系数: {head_info['coefficients']}")

            print(f"\n平均注意力权重分布: {np.mean(analysis['attention_weights'], axis=0)}")


        elif args.model == 'multi_head_contrastive':
            analysis = model.get_detailed_contrastive_analysis(g, features)
        elif args.model == 'multi_head_teacher_student':
            analysis = model.get_detailed_teacher_student_analysis(g, features)

    if args.model in ['multi_head_contrastive', 'multi_head_teacher_student']:
        print(f"头数: {analysis['num_heads']}")
        print(f"专门化分数: {analysis['head_specialization_score']:.3f} (越接近1.0越好)")
        print(f"对比学习温度: {analysis['contrastive_temp']}")

        print("\n各头频段专门化情况:")
        for head_name, head_info in analysis['head_theta_analysis'].items():
            print(f"  {head_name}: {head_info['frequency_focus']}")
            print(f"    系数: {head_info['coefficients']}")

        print(f"\n平均注意力权重分布: {analysis.get('avg_head_weights', 'N/A')}")
        if 'attention_entropy' in analysis:
            print(f"注意力熵 (平均): {analysis['attention_entropy']['average']:.3f}")
            print(f"  解释: {analysis['attention_entropy']['interpretation']}")

    return analysis

if __name__ == "__main__":
    print("正则化对比学习训练模块创建成功！")
    print("1. 支持复合损失：L_total = L_cls + λ_contrast * L_contrastive + λ_div * L_diversity")
    print("2. Teacher-Student对比损失：强制头部专门化")
    print("3. Barlow Twins多样性损失：强制头部表示正交")
    print("4. 多样性损失warm-up机制：前N个epoch逐渐增加权重")
    print("5. 详细的训练过程分析和损失组件跟踪")
    print("6. 正则化头部专门化分析工具")
