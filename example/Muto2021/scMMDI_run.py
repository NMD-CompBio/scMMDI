import os
import os.path as osp
from argparse import ArgumentParser

import scanpy as sc
import mudata as md
import anndata as ad
from scMMDI import scMMDI
from scMMDI.preprocess import merge_obs_from_all_modalities
from scib import metrics as scme
import matplotlib
# matplotlib.use('Agg')  # 如果只保存图片
matplotlib.use('TkAgg')  # 如果需要弹窗查看

def main():
    parser = ArgumentParser()
    parser.add_argument("--preproc_data_dir", default="./res")
    parser.add_argument("--preproc_data_name", default="muto2021_updated")
    parser.add_argument("--results_dir", default="./res")
    parser.add_argument("--results_name", default="muto2021_fit_once")
    parser.add_argument("--shared_feature_keys", default="shared_rna_atac",
                        choices=["shared_rna_atac"],
                        help="Key for shared features in obsm")
    parser.add_argument("--shared_nfeatures", type=int, default=None, help="Number of shared features")
    parser.add_argument("--n_knn", type=int, default=30, help="Number of nearest neighbors for MNN")
    parser.add_argument("--loss_weight_mnn", type=float, default=0.5, help="Weight for MNN loss")
    args = parser.parse_args()

    batch_name, label_name = "batch", "cell_type"
    os.makedirs(args.results_dir, exist_ok=True)
    save_dir = args.results_dir
    save_name = args.results_name

    mdata = md.read(osp.join(
        args.preproc_data_dir, f"{args.preproc_data_name}.h5mu"
    ))
    merge_obs_from_all_modalities(mdata, key=batch_name)
    merge_obs_from_all_modalities(mdata, key=label_name)

    model = scMMDI(
        dim_c=13,
        input_key="lsi_pca",
        net_key="net",
        balance_sample="max",
        num_workers=0,
        batch_key=batch_name,
        batch_size=256,
        disc_gradient_weight=50,
        drop_self_loop=False,
        mix_dec_dot_weight=0.99,
        loss_weight_mnn=0.5,
        loss_weight_disc=1.0,
        loss_weight_kl_omics=0.8,
        p_modal=[1. / 3., 1. / 3., 1. / 3.],
        p_random=0.1,
    )
    model.fit(mdata)
    mdata.obs["scMMDI_c_label"] = mdata.obsm["scMMDI_c"].argmax(axis=1)

    sc.pp.neighbors(mdata, use_rep="scMMDI_z")
    sc.tl.leiden(mdata, resolution=0.1, key_added="leiden")

    model.differential(mdata, "leiden")

    sc.tl.umap(mdata, min_dist=0.2)
    # convert categorical
    mdata.obs[batch_name] = mdata.obs[batch_name].astype("category")
    mdata.obs["scMMDI_c_label"] = mdata.obs["scMMDI_c_label"].astype(
        "category"
    )
    # plot and save umap
    fig_umap = sc.pl.umap(
        mdata,
        color=[batch_name, label_name, "scMMDI_c_label", "leiden"],
        ncols=2,
        return_fig=True,
    )
    fig_umap.savefig(osp.join(save_dir, f"{save_name}.png"))
    mdata.write(osp.join(save_dir, f"{save_name}.h5mu"))

    adata = ad.AnnData(obs=mdata.obs[["leiden", label_name]])
    ari = scme.ari(adata, cluster_key="leiden", label_key=label_name)
    nmi = scme.nmi(adata, cluster_key="leiden", label_key=label_name)
    print(f"ARI = {ari:.4f}, NMI = {nmi:.4f}")


if __name__ == "__main__":
    main()


# # %%
# # running for muto2021
#
#     cfg_cp.weights.update(
#         {
#             "weight_dot": 0.99,
# #           "weight_mlp": 0.01,
# #           "alpha": 50,
#             "loss_weight_contractive_graph": 1.0,
#             "loss_unobs_weight": 0.12 + 0.1 * mask_ratio,
#         }
#     )
#     cfg_cp.train.update(
#         {
#             "early_stop_patient": 20,
#         }
#     )
#
