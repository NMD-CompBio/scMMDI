from typing import (
    Any,
    Dict,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from itertools import combinations
from torch import cdist
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
from torch.nn.functional import pairwise_distance
from .dataset import GMINIBATCH, ModalMaskGenerator
from .encoders import AttVEncoder, AttGMMVEncoder, GraphEncoder
from .decoders import (
    MLPMultiModalDecoder,
    DotMultiModalDecoder,
    MixtureMultiModalDecoder,
    GraphDecoder,
)
from .discriminator import Discriminator
from sklearn.neighbors import NearestNeighbors

T = torch.Tensor
FRES = Dict[str, Any]  # forward results
LOSS = Dict[str, T]


def acquire_pairs(feat1: torch.Tensor, feat2: torch.Tensor, k: int = 30) -> torch.Tensor:
    """
    Compute mutual nearest neighbors between two feature matrices.

    Args:
        feat1: First feature matrix (n_samples, n_features).
        feat2: Second feature matrix (n_samples, n_features).
        k: Number of nearest neighbors to consider.

    Returns:
        sim: Similarity matrix (n_samples, n_samples) indicating MNN pairs (1.0 for MNN, 0.0 otherwise).
    """
    feat1_np = feat1.cpu().numpy()
    feat2_np = feat2.cpu().numpy()

    nn1 = NearestNeighbors(n_neighbors=k).fit(feat2_np)
    nn2 = NearestNeighbors(n_neighbors=k).fit(feat1_np)

    dist1, idx1 = nn1.kneighbors(feat1_np)
    dist2, idx2 = nn2.kneighbors(feat2_np)

    sim = torch.zeros((feat1.shape[0], feat2.shape[0]), dtype=torch.float32, device=feat1.device)
    for i in range(feat1.shape[0]):
        for j in idx1[i]:
            if i in idx2[j]:
                sim[i, j] = 1.0
    return sim

class scMMDINET(nn.Module):
    def __init__(
        self,
        dim_inputs: Dict[str, int],
        dim_outputs: Dict[str, int],
        nbatches: int,
        mixture_embeddings: bool = True,
        decoder_style: Literal["mlp", "glue", "mixture"] = "mixture",
        disc_alignment: bool = True,
        dim_z: int = 30,
        dim_u: int = 30,
        dim_c: int = 6,
        dim_enc_middle: int = 200,
        hiddens_enc_unshared: Sequence[int] = (256, 256),
        hiddens_enc_z: Sequence[int] = (256,),
        hiddens_enc_c: Sequence[int] = (50,),
        hiddens_enc_u: Sequence[int] = (50,),
        hiddens_dec: Optional[Sequence[int]] = (
            256,
            256,
        ),
        hiddens_prior: Sequence[int] = (),
        hiddens_disc: Optional[Sequence[int]] = (256, 256),
        distributions: Union[str, Mapping[str, str]] = "nb",
        distributions_style: Union[str, Mapping[str, str]] = "batch",
        bn: bool = True,
        act: str = "lrelu",
        dp: float = 0.2,
        disc_bn: Optional[bool] = True,
        disc_condi_train: Optional[str] = None,
        disc_on_mean: bool = False,
        disc_criterion: Literal["ce", "bce", "focal"] = "ce",
        c_reparam: bool = True,
        c_prior: Optional[Sequence[float]] = None,
        omic_embed: bool = False,
        omic_embed_train: bool = False,
        spectral_norm: bool = False,
        input_with_batch: bool = False,
        reduction: str = "sum",
        semi_supervised: bool = False,
        graph_encoder_init_zero: bool = True,
        # graph_decoder_whole: bool = False,
        temperature: float = 1.0,
        disc_gradient_weight: float = 50.0,  # alpha
        label_smooth: float = 0.1,
        focal_alpha: float = 2.0,
        focal_gamma: float = 1.0,
        mix_dec_dot_weight: float = 0.9,
        loss_weight_kl_omics: float = 0.8,
        loss_weight_rec_omics: float = 1.0,
        loss_weight_kl_graph: float = 0.01,
        loss_weight_rec_graph: Tuple[float, str] = "nomics",
        loss_weight_disc: float = 1.0,
        loss_weight_sup: float = 1.0,
        mask_embed_dim: int = 16,
        shared_feature_dims: Optional[Dict[str, int]] = None,  # 新增
        loss_weight_mnn: float = 0.5,  # 新增
        n_knn: int = 30,
        p_random: float = 0.08,  # 新增：随机掩码概率
        p_modal: Optional[torch.Tensor] = None,  # 新增：模态缺失概率

    ) -> None:
        assert decoder_style in ["mlp", "glue", "mixture"]
        if loss_weight_rec_graph != "nomics":
            assert isinstance(loss_weight_rec_graph, float)
        else:
            loss_weight_rec_graph = len(dim_outputs)
        super().__init__()

        self.decoder_style = decoder_style
        if disc_alignment:
            self.discriminator = Discriminator(
                inc=dim_z,
                outc=nbatches,
                hiddens=hiddens_disc,
                nclusters=dim_c,
                disc_on_mean=disc_on_mean,
                disc_condi_train=disc_condi_train,
                act=act,
                bn=disc_bn,
                dp=dp,
                criterion=disc_criterion,
                spectral_norm=spectral_norm,
                gradient_reversal=True,
                gradient_alpha=disc_gradient_weight,
                label_smooth=label_smooth,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
            )
        else:
            self.discriminator = None
        if mixture_embeddings:
            self.encoder = AttGMMVEncoder(
                incs=dim_inputs,
                zdim=dim_z,
                cdim=dim_c,
                udim=dim_u,
                nbatch=nbatches if input_with_batch else 0,
                hiddens_unshare=hiddens_enc_unshared,
                nlatent_middle=dim_enc_middle,
                hiddens_z=hiddens_enc_z,
                hiddens_c=hiddens_enc_c,
                hiddens_u=hiddens_enc_u,
                hiddens_prior=hiddens_prior,
                act=act,
                bn=bn,
                dp=dp,
                reduction=reduction,
                omic_embed=omic_embed,
                omic_embed_train=omic_embed_train,
                c_reparam=c_reparam,
                semi_supervised=semi_supervised,
                c_prior=c_prior,
                temperature=temperature,
            )
        else:
            self.encoder = AttVEncoder(
                incs=dim_inputs,
                outc=dim_z,
                nbatch=nbatches if input_with_batch else 0,
                hiddens_unshare=hiddens_enc_unshared,
                nlatent_middle=dim_enc_middle,
                hiddens_shared=hiddens_enc_z,
                act=act,
                bn=bn,
                dp=dp,
                reduction=reduction,
                omic_embed=omic_embed,
                omic_embed_train=omic_embed_train,
            )

        if decoder_style == "mlp":
            self.decoder = MLPMultiModalDecoder(
                inc=dim_z,
                outcs=dim_outputs,
                hiddens=hiddens_dec,
                nbatch=nbatches,
                act=act,
                bn=bn,
                dp=dp,
                reduction=reduction,
                distributions=distributions,
                distributions_style=distributions_style,
            )
        elif decoder_style == "glue":
            # NOTE: hiddens_dec will used in non-linear transformation.
            # when using glue style decoder，shoule set hiddens as None
            self.decoder = DotMultiModalDecoder(
                outcs=dim_outputs,
                nbatch=nbatches,
                inpt=dim_z,
                hiddens=hiddens_dec,
                act=act,
                bn=bn,
                dp=dp,
                reduction=reduction,
                distributions=distributions,
                distributions_style=distributions_style,
            )
        elif decoder_style == "mixture":
            self.decoder = MixtureMultiModalDecoder(
                inc=dim_z,
                outcs=dim_outputs,
                hiddens=hiddens_dec,
                nbatch=nbatches,
                act=act,
                bn=bn,
                dp=dp,
                weight_dot=mix_dec_dot_weight,
                reduction=reduction,
                distributions=distributions,
                distributions_style=distributions_style,
            )

        # glue and mixture will both need graph encoder and decoder
        if decoder_style != "mlp":
            self.gencoder = GraphEncoder(
                dim_inputs,
                sum(dim_outputs.values()),
                dim_z+mask_embed_dim,
                reduction=reduction,
                zero_init=graph_encoder_init_zero,
            )
            self.gdecoder = GraphDecoder()

        # 初始化掩码生成器
        self.mask_generator = ModalMaskGenerator(
            modality_names=list(dim_inputs.keys()),
            p_modal=p_modal,
            p_random=p_random,
            max_p_random=0.2
        )

        self._dim_outputs = dim_outputs  # for trainer
        self._loss_weight_kl_graph = loss_weight_kl_graph
        self._loss_weight_rec_graph = loss_weight_rec_graph
        self._loss_weight_kl_omics = loss_weight_kl_omics
        self._loss_weight_rec_omics = loss_weight_rec_omics
        self._loss_weight_disc = loss_weight_disc
        self._loss_weight_sup = loss_weight_sup
        self._loss_weight_mnn = loss_weight_mnn
        self.shared_feature_dims = shared_feature_dims
        self.n_knn = n_knn


    def forward(self, batch: GMINIBATCH, training: bool = True) -> FRES:

        # 生成随机掩码（仅在训练时）
        if training:
            batch["mask"] = self.mask_generator(batch, device=next(self.parameters()).device)
        enc_res = self.encoder(batch)

        return enc_res

    def reconstruct(
        self, batch: GMINIBATCH, v_dist: Optional[Normal] = None,
        random: bool = True,
    ) -> FRES:
        print("开始数据重建")
        if not random:
            raise NotImplementedError

        # 生成随机掩码（仅在训练时）
        if self.training:
            batch["mask"] = self.mask_generator(batch, device=next(self.parameters()).device)
        enc_res = self.encoder(batch)

        if "zsample" not in enc_res:
            enc_res["zsample"] = enc_res["z"].sample()

        #if "z_unobs_sample" not in enc_res:
            #enc_res["z_unobs_sample"] = enc_res["z_unobs"].sample()
        enc_res["vsample"] = v_dist.sample()
        dec_res = self.decoder(batch, enc_res)
        return dec_res["dist"]

    def step(self, batch: GMINIBATCH) -> Tuple[FRES, FRES, FRES, LOSS]:

        all_loss = {}

        # 生成随机掩码（仅在训练时）
        if self.training:
            batch["mask"] = self.mask_generator(batch, device=next(self.parameters()).device)
            print("使用了随机掩码")
        enc_res, enc_loss = self.encoder.step(batch)

        if "zsample" not in enc_res:
            enc_res["zsample"] = enc_res["z"].rsample()

        #if "z_unobs_sample" not in enc_res:
            #enc_res["z_unobs_sample"] = enc_res["z_unobs"].sample()
        for k, v in enc_loss.items():
            all_loss[f"enc/{k}"] = v

        if self.decoder_style != "mlp":
            genc_res, genc_loss = self.gencoder.step(batch)
            enc_res.update(genc_res)
            for k, v in genc_loss.items():
                all_loss[f"genc/{k}"] = v

        dec_res, dec_loss = self.decoder.step(batch, enc_res)
        for k, v in dec_loss.items():
            all_loss[f"dec/{k}"] = v

        if self.discriminator is not None:
            disc_res, disc_loss = self.discriminator.step(
                batch, enc_res, dec_res
            )
            for k, v in disc_loss.items():
                all_loss[f"disc/{k}"] = v
        else:
            disc_res = None

            # 计算 MNN 损失

        mnn_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        z = enc_res["zsample"]
        if self.shared_feature_dims and 'shared_features' in batch:
            shared_features = batch["shared_features"]
            for shared_key in self.shared_feature_dims:
                modality_pairs = []
                if shared_key == 'shared_rna_adt' and 'rna' in shared_features and 'protein' in shared_features:
                    modality_pairs.append(('rna', 'protein'))
                elif shared_key == 'shared_rna_atac' and 'rna' in shared_features and 'atac' in shared_features:
                    modality_pairs.append(('rna', 'atac'))
                for k1, k2 in modality_pairs:
                    mask1 = batch["mask"][k1]
                    mask2 = batch["mask"][k2]
                    mask1_bool = (mask1 == 0.0)
                    mask2_bool = (mask2 == 0.0)
                    valid_mask = mask1_bool & mask2_bool
                    if not valid_mask.any():
                        if not valid_mask.any():
                            print(f"No valid samples for {shared_key} ({k1}, {k2})")
                        continue

                    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                    batch_mask = batch["blabel"][valid_indices] == batch["blabel"][valid_indices][0]  # 同一批次
                    if batch_mask.any():
                        z1 = z[valid_indices][batch_mask]
                        z2 = z[valid_indices][batch_mask]
                        x_p1 = shared_features[k1][shared_key][valid_indices][batch_mask]
                        x_p2 = shared_features[k2][shared_key][valid_indices][batch_mask]
                        #if not valid_mask.any():
                            #continue
                        #valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                        #if len(valid_indices) < self.n_knn:
                            #self.n_knn = max(20, len(valid_indices) // 2)  # 动态调整 n_knn
                        if x_p1.size(0) < 2 or x_p2.size(0) < 2:
                            print(f"Skipping MNN for {k1}-{k2}: insufficient samples ({x_p1.size(0)}).")
                            continue
                            #动态调整 n_knn
                        effective_samples = min(x_p1.size(0), x_p2.size(0))
                        n_knn = min(self.n_knn, effective_samples )  # 确保 n_knn <= n_samples - 1
                        if n_knn < 1:
                            print(f"Skipping MNN for {k1}-{k2}: n_knn ({n_knn}) too small.")
                            continue

                    sim = acquire_pairs(x_p1, x_p2, k=n_knn)
                    cosk = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)
                    cosk_p = F.cosine_similarity(x_p1.unsqueeze(1), x_p2.unsqueeze(0), dim=-1)
                    mnn_loss += torch.sum(sim * (cosk_p - cosk).pow(2)) / (torch.sum(sim) + 1e-6)
                all_loss["mnn"] = mnn_loss * self._loss_weight_mnn
                print(f"MNN Loss: {mnn_loss.item()}")

        z = enc_res["zsample"]
        mask_embed = enc_res.get("mask_embed")
        z_extended = torch.cat([z, mask_embed], dim=-1) if mask_embed is not None else z
        v = genc_res["vsample"]

        # 归一化表示
        z_extended = F.normalize(z_extended, dim=-1)
        v = F.normalize(v, dim=-1)

        # 检查设备一致性
        device = z_extended.device
        assert v.device == device, "z_extended and v must be on the same device"

        contrast_loss = torch.tensor(0.0, device=device)
        normed_subgraph = batch.get("normed_subgraph", None)
        if normed_subgraph is not None and len(normed_subgraph) >= 5:
            # 解包 normed_subgraph
            row_idx, col_idx, es, ew, ew_norm = normed_subgraph
            row_idx = torch.tensor(row_idx, device=device, dtype=torch.long)
            col_idx = torch.tensor(col_idx, device=device, dtype=torch.long)
            es = torch.tensor(es, device=device, dtype=torch.float32)
            ew = torch.tensor(ew, device=device, dtype=torch.float32)
            ew_norm = torch.tensor(ew_norm, device=device, dtype=torch.float32)

            # 使用 ew 区分正负样本
            pos_mask = ew == 1.0  # 正样本边 (ew == 1.0, es == 1.0)
            neg_mask = ew == 0.0  # 负样本边 (ew == 0.0, es == -1.0)

            # 提取正样本对
            idx1 = row_idx[pos_mask]
            idx2 = col_idx[pos_mask]
            pos_ew_norm = ew_norm[pos_mask]

            # 提取负样本对
            idx1_neg = row_idx[neg_mask]
            idx2_neg = col_idx[neg_mask]

            if len(idx1) == 0:
                print("No valid positive pairs in normed_subgraph (ew == 1.0)")
            elif len(idx1_neg) == 0:
                print("No valid negative pairs in normed_subgraph (ew == 0.0)")
            else:
                num_feats = v.size(0)


                # 计算特征-特征相似度
                f = lambda x: torch.exp(x / 1.0)
                sim = torch.matmul(v, v.T) / 1.0  # [num_feats, num_feats]
                exp_sim = f(sim)

                # 正样本相似度（使用归一化边权重加权）
                pos_sim = sim[idx1, idx2] * pos_ew_norm
                exp_pos_sim = f(pos_sim)

                # 负样本相似度
                neg_sim = sim[idx1_neg, idx2_neg]
                exp_neg_sim = f(neg_sim)

                # 计算对比损失（InfoNCE形式）
                denom = exp_pos_sim + exp_neg_sim.sum(dim=0, keepdim=True)
                contrast_loss = -torch.log(exp_pos_sim / (denom + 1e-6)).mean()

        else:


        all_loss["contrast_graph"] = contrast_loss * 1.0


        metric_loss = 0.0
        for k, v in all_loss.items():
            if k.startswith("disc"):
                continue
            elif "kl_graph" in k:
                metric_loss += v * self._loss_weight_kl_graph
            elif "rec_graph" in k:
                metric_loss += v * self._loss_weight_rec_graph
            elif k == "contrast_graph":
                metric_loss += v
            elif k.startswith("dec"):
                metric_loss += v * self._loss_weight_rec_omics
            elif k.startswith("enc") and ("sup" in k):
                metric_loss += v * self._loss_weight_sup
            elif k.startswith("enc") and ("kl" in k):
                metric_loss += v * self._loss_weight_kl_omics
            elif k == "mnn":
                metric_loss += v
            else:
                metric_loss += v
        all_loss["metric"] = metric_loss
        total_loss = metric_loss.clone()
        for k, v in all_loss.items():
            if k.startswith("disc"):
                total_loss += v * self._loss_weight_disc
        all_loss["total"] = total_loss

        return enc_res, dec_res, disc_res, all_loss
