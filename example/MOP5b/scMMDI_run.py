import os
import os.path as osp
from argparse import ArgumentParser

import mudata as md
import scanpy as sc
import anndata as ad
from scMMDI import scMMDI
from scMMDI.preprocess import merge_obs_from_all_modalities
from scib import metrics as scme
import matplotlib


def fit_once(
    mdata: md.MuData,
    save_dir: str,
    save_name: str,
    print_metrics: bool = True,
    label_name: str = "cell_type",
) -> None:
    model = scMMDI(
        dim_c=21,
        input_key="log1p_norm",
        net_key="net",
        balance_sample="max",
        num_workers=0,
        loss_weight_mnn=0.5,
        loss_weight_disc=0.6,
        loss_weight_kl_omics=1.0,
        p_modal=[1. / 3., 1. / 3., 1. / 3.],
        p_random=0.05,
    )
    model.fit(mdata)
    mdata.obs["scMMDI_c_label"] = mdata.obsm["scMMDI_c"].argmax(axis=1)

    sc.pp.neighbors(mdata, use_rep="scMMDI_z")
    sc.tl.leiden(mdata, resolution=0.1, key_added="leiden")

    model.differential(mdata, "leiden")

    sc.tl.umap(mdata, min_dist=0.2)
    # convert categorical
    mdata.obs["batch"] = mdata.obs["batch"].astype("category")
    mdata.obs["scMMDI_c_label"] = mdata.obs["scMMDI_c_label"].astype(
        "category"
    )

    mdata.write(osp.join(save_dir, f"{save_name}.h5mu"))
    # plot and save umap
    fig_umap = sc.pl.umap(
        mdata,
        color=["batch", label_name, "scMMDI_c_label", "leiden"],
        ncols=2,
        return_fig=True,
    )
    fig_umap.savefig(osp.join(save_dir, f"{save_name}.png"))
    
    if print_metrics:
        adata = ad.AnnData(obs=mdata.obs[["leiden", label_name]])
        ari = scme.ari(adata, cluster_key="leiden", label_key=label_name)
        nmi = scme.nmi(adata, cluster_key="leiden", label_key=label_name)
        print(f"ARI = {ari:.4f}, NMI = {nmi:.4f}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--preproc_data_dir", default="./res")
    parser.add_argument("--preproc_data_name", default="mop5b_updated")
    parser.add_argument("--results_dir", default="./res")
    parser.add_argument("--results_name", default="mop5b_fit_once")
    parser.add_argument("--shared_feature_keys", default="shared_rna_atac", choices=["shared_rna_atac"],
                        help="Key for shared features in obsm")
    parser.add_argument("--shared_nfeatures", type=int, default=None, help="Number of shared features")
    parser.add_argument("--n_knn", type=int, default=30, help="Number of nearest neighbors for MNN")
    parser.add_argument("--loss_weight_mnn", type=float, default=0.5, help="Weight for MNN loss")
    args = parser.parse_args()
    args = parser.parse_args()

    mdata_fn = osp.join(
        args.preproc_data_dir, f"{args.preproc_data_name}.h5mu"
    )
    os.makedirs(args.results_dir, exist_ok=True)

    mdata = md.read(mdata_fn)
    merge_obs_from_all_modalities(mdata, key="cell_type")
    merge_obs_from_all_modalities(mdata, key="batch")
    print(mdata)

    fit_once(mdata, args.results_dir, args.results_name)


if __name__ == "__main__":
    main()

# # %%
# # running for MOP5b
#
#     cfg_cp.weights.update(
#         {
#             "weight_dot": 0.90,
#             "weight_mlp": 0.10,
#             "loss_weight_contractive_graph": 0.5,
#             "loss_unobs_weight": 0.1 + 0.1 * mask_ratio,
#         }
#     )
#     cfg_cp.train.update(
#         {
#             "early_stop_patient": 10,
#         }
#     )
#
