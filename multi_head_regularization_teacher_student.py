import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
from scipy.stats import skew, kurtosis

def _to_scipy_csr_adj(graph: dgl.DGLGraph) -> sp.csr_matrix:

    g_homo = dgl.to_homogeneous(graph) if isinstance(graph, dgl.DGLHeteroGraph) else graph

    try:
        adj = g_homo.adjacency_matrix()
        if hasattr(adj, 'to_scipy'):
            return adj.to_scipy('csr')
    except Exception:
        pass

    try:
        adj = g_homo.adjacency_matrix()
        if hasattr(adj, 'to_sparse_csr'):
            adj_csr = adj.to_sparse_csr()
            return sp.csr_matrix((adj_csr.val.cpu().numpy(),
                                adj_csr.indices().cpu().numpy(),
                                adj_csr.indptr().cpu().numpy()),
                               shape=adj_csr.shape)
    except Exception:
        pass

    try:
        adj = g_homo.adjacency_matrix()
        if hasattr(adj, 'coalesce'):
            adj = adj.coalesce()

        if hasattr(adj, 'values'):
            values = adj.values().cpu().numpy()
        elif hasattr(adj, 'val'):
            values = adj.val.cpu().numpy()
        else:
            raise AttributeError("Cannot find values/val attribute")

        if hasattr(adj, 'indices'):
            indices = adj.indices().cpu().numpy()
        elif hasattr(adj, 'idx'):
            indices = adj.idx.cpu().numpy()
        else:
            raise AttributeError("Cannot find indices/idx attribute")

        shape = adj.shape
        return sp.csr_matrix((values, (indices[0], indices[1])), shape=shape)
    except Exception:
        pass

    try:
        adj = g_homo.adjacency_matrix()
        adj_dense = adj.to_dense().cpu().numpy()
        return sp.csr_matrix(adj_dense)
    except Exception as e:
        raise RuntimeError(f"Failed to convert DGL adjacency matrix to scipy CSR: {e}")

def _build_normalized_laplacian_csr(graph: dgl.DGLGraph) -> sp.csr_matrix:

    adj_csr = _to_scipy_csr_adj(graph)
    L = sp.csgraph.laplacian(adj_csr, normed=True)
    return L.tocsr()

def _csr_to_torch_sparse_coo(L: sp.csr_matrix, device: torch.device) -> torch.Tensor:

    coo = L.tocoo()
    indices = np.vstack((coo.row, coo.col))
    i = torch.from_numpy(indices).long()
    v = torch.from_numpy(coo.data).float()
    return torch.sparse_coo_tensor(i, v, coo.shape, device=device)

def _compute_structural_eig_stats_for_batch(L: sp.csr_matrix, k: int = 6) -> np.ndarray:

    n = L.shape[0]
    if n < 3:
        eigs = np.linalg.eigvalsh(L.toarray())
    else:
        k_actual = max(1, min(k, n - 2))
        try:
            evals_lm = linalg.eigsh(L, k=k_actual, which='LM', return_eigenvectors=False)
            evals_sm = linalg.eigsh(L, k=k_actual, which='SM', return_eigenvectors=False)
            eigs = np.unique(np.concatenate([evals_lm, evals_sm]))
        except Exception:
            eigs = np.linalg.eigvalsh(L.toarray())
    eigs = np.real(eigs)
    return np.array([
        np.mean(eigs),
        np.var(eigs),
        skew(eigs),
        kurtosis(eigs)
    ], dtype=np.float32)

class MultiHeadParamNet_Chebyshev(nn.Module):

    def __init__(self, input_dim: int, poly_order: int, num_heads: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.poly_order = poly_order
        self.num_heads = num_heads
        self.output_dim = num_heads * (poly_order + 1)

        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.multi_head_output = nn.Linear(hidden_dim, self.output_dim)

        self.head_specialization = nn.ModuleList([
            nn.Linear(hidden_dim, poly_order + 1) for _ in range(num_heads)
        ])

        self.use_specialization = True

    def forward(self, spectral_fingerprint: torch.Tensor) -> torch.Tensor:

        if spectral_fingerprint.dim() == 1:
            spectral_fingerprint = spectral_fingerprint.unsqueeze(0)

        shared_features = self.shared_layers(spectral_fingerprint)

        if self.use_specialization:

            multi_head_coeffs = []
            for head_layer in self.head_specialization:
                head_coeffs = head_layer(shared_features)
                multi_head_coeffs.append(head_coeffs.squeeze(0))

            return torch.stack(multi_head_coeffs, dim=0)
        else:

            all_coeffs = self.multi_head_output(shared_features).squeeze(0)
            return all_coeffs.view(self.num_heads, self.poly_order + 1)

class BarlowTwinsDiversityLoss(nn.Module):

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, head_outputs: list) -> torch.Tensor:

        if len(head_outputs) < 2:
            return torch.tensor(0.0, device=head_outputs[0].device)

        stacked_heads = torch.stack(head_outputs, dim=1)
        N, K, d_out = stacked_heads.shape

        total_diversity_loss = 0.0

        for n in range(N):

            node_heads = stacked_heads[n]

            normalized_heads = F.normalize(node_heads, p=2, dim=1, eps=self.eps)

            cross_corr_matrix = torch.mm(normalized_heads, normalized_heads.t())

            identity_matrix = torch.eye(K, device=cross_corr_matrix.device)

            diff_matrix = cross_corr_matrix - identity_matrix
            diversity_loss_node = torch.sum(diff_matrix ** 2)

            total_diversity_loss += diversity_loss_node

        return total_diversity_loss / N

    def get_diversity_analysis(self, head_outputs: list) -> dict:

        if len(head_outputs) < 2:
            return {'error': 'Need at least 2 heads for diversity analysis'}

        stacked_heads = torch.stack(head_outputs, dim=1)
        N, K, d_out = stacked_heads.shape

        avg_cross_corr = torch.zeros(K, K, device=stacked_heads.device)

        for n in range(N):
            node_heads = stacked_heads[n]
            normalized_heads = F.normalize(node_heads, p=2, dim=1, eps=self.eps)
            cross_corr = torch.mm(normalized_heads, normalized_heads.t())
            avg_cross_corr += cross_corr

        avg_cross_corr /= N

        identity_matrix = torch.eye(K, device=avg_cross_corr.device)
        off_diagonal_sum = torch.sum(torch.abs(avg_cross_corr - identity_matrix))
        diagonal_deviation = torch.sum(torch.abs(torch.diag(avg_cross_corr) - 1.0))

        return {
            'avg_cross_correlation_matrix': avg_cross_corr.cpu().numpy(),
            'off_diagonal_sum': off_diagonal_sum.item(),
            'diagonal_deviation': diagonal_deviation.item(),
            'diversity_score': 1.0 / (1.0 + off_diagonal_sum.item()),
            'num_heads': K,
            'feature_dim': d_out
        }

class TeacherStudentContrastiveLoss(nn.Module):

    def __init__(self, feature_dim: int, temperature: float = 0.1, momentum: float = 0.999):
        super().__init__()
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.momentum = momentum

        self.register_buffer('teacher_representation', torch.zeros(feature_dim))
        self.register_buffer('teacher_initialized', torch.tensor(False))

        self.use_projection = True
        if self.use_projection:
            self.teacher_projection = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, feature_dim)
            )
            self.student_projection = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, feature_dim)
            )

    def update_teacher(self, current_fused_output: torch.Tensor):

        with torch.no_grad():

            current_mean = current_fused_output.mean(dim=0)

            if not self.teacher_initialized:

                self.teacher_representation.copy_(current_mean)
                self.teacher_initialized.fill_(True)
            else:

                self.teacher_representation.mul_(self.momentum).add_(
                    current_mean, alpha=1 - self.momentum
                )

    def info_nce_loss(self, anchor: torch.Tensor, positive: torch.Tensor,
                     negatives: torch.Tensor) -> torch.Tensor:

        pos_sim = F.cosine_similarity(anchor, positive, dim=-1)
        pos_sim = pos_sim / self.temperature

        neg_sim = F.cosine_similarity(
            anchor.unsqueeze(1), negatives, dim=-1
        )
        neg_sim = neg_sim / self.temperature

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)

    def forward(self, head_outputs: list, current_fused_output: torch.Tensor) -> torch.Tensor:

        self.update_teacher(current_fused_output)

        N = head_outputs[0].shape[0]
        teacher_positive = self.teacher_representation.unsqueeze(0).expand(N, -1)

        if self.use_projection:
            teacher_positive = self.teacher_projection(teacher_positive)
            projected_heads = [self.student_projection(head) for head in head_outputs]
        else:
            projected_heads = head_outputs

        num_heads = len(projected_heads)
        total_loss = 0.0

        for i, head_i in enumerate(projected_heads):

            positive = teacher_positive

            negatives = []
            for j, head_j in enumerate(projected_heads):
                if i != j:
                    negatives.append(head_j)

            if len(negatives) > 0:
                negatives = torch.stack(negatives, dim=1)

                loss_i = self.info_nce_loss(head_i, positive, negatives)
                total_loss += loss_i

        return total_loss / num_heads

    def get_teacher_info(self) -> dict:

        return {
            'teacher_initialized': self.teacher_initialized.item(),
            'teacher_norm': torch.norm(self.teacher_representation).item(),
            'momentum': self.momentum,
            'temperature': self.temperature
        }

class RegularizedTeacherStudentChannelWiseAttentionFusion(nn.Module):

    def __init__(self, d_out: int, num_heads: int, hidden_dim: int = 32,
                 contrastive_temp: float = 0.1, teacher_momentum: float = 0.999):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads

        self.attn_mlp = nn.Sequential(
            nn.Linear(d_out, hidden_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        self.use_positional_encoding = True
        if self.use_positional_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(num_heads, d_out) * 0.1)

        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.use_gating = True
        if self.use_gating:
            self.gate_mlp = nn.Sequential(
                nn.Linear(d_out * num_heads, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

        self.teacher_student_contrastive = TeacherStudentContrastiveLoss(
            feature_dim=d_out,
            temperature=contrastive_temp,
            momentum=teacher_momentum
        )

        self.diversity_loss_module = BarlowTwinsDiversityLoss()

    def forward(self, heads_outputs: list, return_losses: bool = False) -> tuple:

        stacked_heads = torch.stack(heads_outputs, dim=1)
        N, K, d_out = stacked_heads.shape

        if self.use_positional_encoding:
            pos_encoded = stacked_heads + self.pos_encoding.unsqueeze(0)
        else:
            pos_encoded = stacked_heads

        reshaped_heads = pos_encoded.view(N * K, d_out)
        scores = self.attn_mlp(reshaped_heads)
        scores = scores.view(N, K)

        scaled_scores = scores / self.temperature
        weights = F.softmax(scaled_scores, dim=1)

        weights_expanded = weights.unsqueeze(-1)
        weighted_heads = stacked_heads * weights_expanded
        fused_output = torch.sum(weighted_heads, dim=1)

        if self.use_gating:
            concat_heads = stacked_heads.view(N, K * d_out)
            gate = self.gate_mlp(concat_heads)
            avg_output = torch.mean(stacked_heads, dim=1)
            fused_output = gate * fused_output + (1 - gate) * avg_output

        losses = {}
        if return_losses:

            contrastive_loss = self.teacher_student_contrastive(heads_outputs, fused_output)
            losses['contrastive_loss'] = contrastive_loss

            diversity_loss = self.diversity_loss_module(heads_outputs)
            losses['diversity_loss'] = diversity_loss

        if return_losses:
            return fused_output, losses
        else:
            return fused_output

    def get_attention_weights(self, heads_outputs: list) -> torch.Tensor:

        with torch.no_grad():
            stacked_heads = torch.stack(heads_outputs, dim=1)
            N, K, d_out = stacked_heads.shape

            if self.use_positional_encoding:
                pos_encoded = stacked_heads + self.pos_encoding.unsqueeze(0)
            else:
                pos_encoded = stacked_heads

            reshaped_heads = pos_encoded.view(N * K, d_out)
            scores = self.attn_mlp(reshaped_heads)
            scores = scores.view(N, K)

            scaled_scores = scores / self.temperature
            weights = F.softmax(scaled_scores, dim=1)

            return weights

    def get_comprehensive_analysis(self, heads_outputs: list) -> dict:

        teacher_student_info = self.teacher_student_contrastive.get_teacher_info()
        diversity_analysis = self.diversity_loss_module.get_diversity_analysis(heads_outputs)

        return {
            'teacher_student_info': teacher_student_info,
            'diversity_analysis': diversity_analysis,
            'attention_weights': self.get_attention_weights(heads_outputs).cpu().numpy(),
            'fusion_mechanism': {
                'use_positional_encoding': self.use_positional_encoding,
                'use_gating': self.use_gating,
                'temperature': self.temperature.item()
            }
        }

class MultiHeadSAConvRegularizedTeacherStudent(nn.Module):

    def __init__(self, in_feats: int, out_feats: int, poly_order: int, num_heads: int = 3,
                 proj_dim: int = 16, fusion_hidden_dim: int = 32, contrastive_temp: float = 0.1,
                 teacher_momentum: float = 0.999):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.poly_order = poly_order
        self.num_heads = num_heads

        proj_matrix = torch.randn(in_feats, proj_dim)
        self.register_buffer('proj_matrix', proj_matrix)

        self.multi_head_param_net = MultiHeadParamNet_Chebyshev(
            input_dim=20,
            poly_order=poly_order,
            num_heads=num_heads
        )

        self.regularized_teacher_student_fusion = RegularizedTeacherStudentChannelWiseAttentionFusion(
            d_out=in_feats,
            num_heads=num_heads,
            hidden_dim=fusion_hidden_dim,
            contrastive_temp=contrastive_temp,
            teacher_momentum=teacher_momentum
        )

        self.final_linear = nn.Linear(in_feats, out_feats)

    def forward(self, g: dgl.DGLGraph, x: torch.Tensor, return_losses: bool = False,
                return_head_outputs: bool = False) -> tuple:

        L_csr = _build_normalized_laplacian_csr(g)
        L_sparse = _csr_to_torch_sparse_coo(L_csr, device=x.device)

        struct_fp_np = _compute_structural_eig_stats_for_batch(L_csr, k=6)
        struct_fp = torch.from_numpy(struct_fp_np).to(x.device)

        if x.shape[1] > self.proj_matrix.shape[1]:
            x_proj = x @ self.proj_matrix.to(x.device)
        else:
            x_proj = x
        Lx = torch.sparse.mm(L_sparse, x_proj)
        xT_Lx = torch.sum(x_proj * Lx, dim=0)
        xT_x = torch.sum(x_proj * x_proj, dim=0) + 1e-8
        rq_vec = xT_Lx / xT_x
        if rq_vec.numel() < 16:
            pad = torch.zeros(16 - rq_vec.numel(), device=x.device)
            signal_fp = torch.cat([rq_vec, pad], dim=0)
        else:
            signal_fp = rq_vec[:16]

        spectral_fp = torch.cat([struct_fp, signal_fp], dim=0)
        multi_head_thetas = self.multi_head_param_net(spectral_fp)

        lambda_max = self._estimate_lambda_max(g)

        def rescaled_laplacian_op(feat: torch.Tensor) -> torch.Tensor:
            g_homo = dgl.to_homogeneous(g) if isinstance(g, dgl.DGLHeteroGraph) else g
            deg = g_homo.out_degrees().float().clamp(min=1).to(feat.device)
            deg_inv_sqrt = torch.pow(deg, -0.5).unsqueeze(-1)
            with g_homo.local_scope():
                g_homo.ndata['h_temp'] = feat * deg_inv_sqrt
                g_homo.update_all(fn.copy_u('h_temp', 'm'), fn.sum('m', 'h_temp'))
                normalized_feat = g_homo.ndata.pop('h_temp') * deg_inv_sqrt
            L_feat = feat - normalized_feat
            return (2.0 / lambda_max) * L_feat - feat

        head_outputs = []
        for head_idx in range(self.num_heads):
            thetas_c = multi_head_thetas[head_idx]

            T0_x = x
            T1_x = rescaled_laplacian_op(x)
            h_agg = thetas_c[0] * T0_x
            if self.poly_order >= 1:
                h_agg = h_agg + thetas_c[1] * T1_x
            for k in range(2, self.poly_order + 1):
                T2_x = 2 * rescaled_laplacian_op(T1_x) - T0_x
                h_agg = h_agg + thetas_c[k] * T2_x
                T0_x, T1_x = T1_x, T2_x

            head_outputs.append(h_agg)

        if return_losses:
            h_fused, losses = self.regularized_teacher_student_fusion(
                head_outputs, return_losses=True
            )
        else:
            h_fused = self.regularized_teacher_student_fusion(
                head_outputs, return_losses=False
            )
            losses = {}

        h_final = self.final_linear(h_fused)
        h_final = F.leaky_relu(h_final)

        result = [h_final]
        if return_losses:
            result.append(losses)
        if return_head_outputs:
            result.append(head_outputs)

        return tuple(result) if len(result) > 1 else h_final

    def _estimate_lambda_max(self, g: dgl.DGLGraph) -> float:

        try:
            if isinstance(g, dgl.DGLHeteroGraph):
                g_homo = dgl.to_homogeneous(g)
                return float(dgl.laplacian_lambda_max(g_homo)[0])
            else:
                return float(dgl.laplacian_lambda_max(g)[0])
        except Exception:
            return 2.0

    def get_comprehensive_regularization_analysis(self, g: dgl.DGLGraph, x: torch.Tensor):

        with torch.no_grad():

            _, _, head_outputs = self.forward(g, x, return_losses=False, return_head_outputs=True)

            comprehensive_analysis = self.regularized_teacher_student_fusion.get_comprehensive_analysis(head_outputs)

            head_similarities = torch.zeros(self.num_heads, self.num_heads)
            for i in range(self.num_heads):
                for j in range(self.num_heads):
                    if i != j:
                        sim = F.cosine_similarity(
                            head_outputs[i].mean(dim=0),
                            head_outputs[j].mean(dim=0),
                            dim=0
                        )
                        head_similarities[i, j] = sim.item()

            L_csr = _build_normalized_laplacian_csr(g)
            struct_fp_np = _compute_structural_eig_stats_for_batch(L_csr, k=6)
            struct_fp = torch.from_numpy(struct_fp_np).to(x.device)

            if x.shape[1] > self.proj_matrix.shape[1]:
                x_proj = x @ self.proj_matrix.to(x.device)
            else:
                x_proj = x
            L_sparse = _csr_to_torch_sparse_coo(L_csr, device=x.device)
            Lx = torch.sparse.mm(L_sparse, x_proj)
            xT_Lx = torch.sum(x_proj * Lx, dim=0)
            xT_x = torch.sum(x_proj * x_proj, dim=0) + 1e-8
            rq_vec = xT_Lx / xT_x
            if rq_vec.numel() < 16:
                pad = torch.zeros(16 - rq_vec.numel(), device=x.device)
                signal_fp = torch.cat([rq_vec, pad], dim=0)
            else:
                signal_fp = rq_vec[:16]

            spectral_fp = torch.cat([struct_fp, signal_fp], dim=0)
            multi_head_thetas = self.multi_head_param_net(spectral_fp)

            return {
                'head_similarities': head_similarities.cpu().numpy(),
                'head_specialization_score': 1.0 - head_similarities.mean().item(),
                'spectral_fingerprint': spectral_fp.cpu().numpy(),
                'multi_head_thetas': multi_head_thetas.cpu().numpy(),
                'num_heads': self.num_heads,
                'head_theta_analysis': {
                    f'head_{i}': {
                        'coefficients': thetas_c.cpu().numpy(),
                        'frequency_focus': self._analyze_frequency_focus(thetas_c.cpu().numpy())
                    }
                    for i, thetas_c in enumerate(multi_head_thetas)
                },
                'regularization_advantages': {
                    'logical_consistency': "Teacher表示通过EMA独立更新，避免了h_fused和h_k的循环依赖",
                    'diversity_regularization': "Barlow Twins启发的多样性损失确保头部表示正交性",
                    'combined_effect': "对比学习 + 多样性正则化 = 更强的头部专门化"
                },
                **comprehensive_analysis
            }

    def _analyze_frequency_focus(self, coeffs: np.ndarray) -> str:

        if len(coeffs) < 2:
            return "unknown"

        if coeffs[0] > 0.5 * np.sum(np.abs(coeffs)):
            return "low_frequency"
        elif coeffs[-1] > 0.3 * np.sum(np.abs(coeffs)):
            return "high_frequency"
        else:
            return "mid_frequency"

class MultiHeadBWGNN_SA_RegularizedTeacherStudent(nn.Module):

    def __init__(self, in_feats: int, h_feats: int, num_classes: int, d: int = 2,
                 num_heads: int = 3, proj_dim: int = 16, fusion_hidden_dim: int = 32,
                 contrastive_temp: float = 0.1, contrastive_weight: float = 0.1,
                 diversity_weight: float = 0.05, teacher_momentum: float = 0.999):
        super().__init__()
        self.num_heads = num_heads
        self.contrastive_weight = contrastive_weight
        self.diversity_weight = diversity_weight
        self.teacher_momentum = teacher_momentum

        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)

        self.multi_head_regularized_conv = MultiHeadSAConvRegularizedTeacherStudent(
            h_feats, h_feats, poly_order=d, num_heads=num_heads,
            proj_dim=proj_dim, fusion_hidden_dim=fusion_hidden_dim,
            contrastive_temp=contrastive_temp, teacher_momentum=teacher_momentum
        )

        self.linear3 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()

        self.use_residual = True
        if self.use_residual:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, g: dgl.DGLGraph, in_feat: torch.Tensor, return_losses: bool = False,
                return_head_outputs: bool = False) -> tuple:

        h = self.linear1(in_feat)
        h = self.act(h)
        h_intermediate = self.linear2(h)
        h = self.act(h_intermediate)

        conv_result = self.multi_head_regularized_conv(
            g, h, return_losses=return_losses, return_head_outputs=return_head_outputs
        )

        if return_losses and return_head_outputs:
            h_filtered, losses, head_outputs = conv_result
        elif return_losses:
            h_filtered, losses = conv_result
            head_outputs = None
        elif return_head_outputs:
            h_filtered, head_outputs = conv_result
            losses = {}
        else:
            h_filtered = conv_result
            losses = {}
            head_outputs = None

        if self.use_residual:
            h_filtered = h_filtered + self.residual_weight * h_intermediate

        logits = self.linear3(h_filtered)

        result = [logits]
        if return_losses:
            result.append(losses)
        if return_head_outputs:
            result.append(head_outputs)

        return tuple(result) if len(result) > 1 else logits

    def compute_total_loss(self, logits: torch.Tensor, labels: torch.Tensor,
                          losses: dict, class_weight: torch.Tensor = None) -> torch.Tensor:

        if class_weight is not None:
            classification_loss = F.cross_entropy(logits, labels, weight=class_weight)
        else:
            classification_loss = F.cross_entropy(logits, labels)

        total_loss = classification_loss

        if 'contrastive_loss' in losses:
            total_loss += self.contrastive_weight * losses['contrastive_loss']

        if 'diversity_loss' in losses:
            total_loss += self.diversity_weight * losses['diversity_loss']

        return total_loss

    def get_detailed_regularization_analysis(self, g: dgl.DGLGraph, in_feat: torch.Tensor):

        with torch.no_grad():
            h = self.linear1(in_feat)
            h = self.act(h)
            h = self.linear2(h)
            h = self.act(h)

            analysis = self.multi_head_regularized_conv.get_comprehensive_regularization_analysis(g, h)

            analysis.update({
                'input_features_shape': in_feat.shape,
                'processed_features_shape': h.shape,
                'contrastive_weight': self.contrastive_weight,
                'diversity_weight': self.diversity_weight,
                'model_info': {
                    'num_heads': self.num_heads,
                    'use_residual': self.use_residual,
                    'residual_weight': self.residual_weight.item() if self.use_residual else 0,
                    'teacher_momentum': self.teacher_momentum
                },
                'training_insights': {
                    'regularization_architecture':
                        "Teacher-Student对比学习 + Barlow Twins多样性正则化双重约束",
                    'head_specialization_interpretation':
                        f"专门化分数: {analysis['head_specialization_score']:.3f} "
                        f"(越接近1.0表示头部越专门化，越接近0.0表示头部越相似)",
                    'diversity_regularization_effect':
                        "Barlow Twins损失强制头部表示正交，避免冗余学习",
                    'contrastive_learning_effect':
                        "Teacher-Student对比学习通过(h_k, h_teacher)正样本对和(h_k_i, h_k_j)负样本对强制头部专门化",
                    'combined_regularization_benefits':
                        "双重正则化确保头部既专门化又正交，提供更强的表达能力和鲁棒性"
                }
            })

            return analysis

def create_multi_head_regularized_teacher_student_model(
    in_feats: int, h_feats: int = 64, num_classes: int = 2, d: int = 2,
    num_heads: int = 3, proj_dim: int = 16, fusion_hidden_dim: int = 32,
    contrastive_temp: float = 0.1, contrastive_weight: float = 0.1,
    diversity_weight: float = 0.05, teacher_momentum: float = 0.999
):

    return MultiHeadBWGNN_SA_RegularizedTeacherStudent(
        in_feats=in_feats,
        h_feats=h_feats,
        num_classes=num_classes,
        d=d,
        num_heads=num_heads,
        proj_dim=proj_dim,
        fusion_hidden_dim=fusion_hidden_dim,
        contrastive_temp=contrastive_temp,
        contrastive_weight=contrastive_weight,
        diversity_weight=diversity_weight,
        teacher_momentum=teacher_momentum
    )

if __name__ == "__main__":

    model = create_multi_head_regularized_teacher_student_model(
        in_feats=100,
        h_feats=64,
        num_classes=2,
        d=2,
        num_heads=3,
        contrastive_temp=0.1,
        contrastive_weight=0.1,
        diversity_weight=0.05,
        teacher_momentum=0.999
    )
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    teacher_student_params = sum(p.numel() for p in
                               model.multi_head_regularized_conv.regularized_teacher_student_fusion.teacher_student_contrastive.parameters())
    diversity_params = sum(p.numel() for p in
                         model.multi_head_regularized_conv.regularized_teacher_student_fusion.diversity_loss_module.parameters())

    print(f"Teacher-Student对比学习模块参数量: {teacher_student_params:,}")
    print(f"Barlow Twins多样性正则化模块参数量: {diversity_params:,}")
