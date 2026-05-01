import torch
from typing import Tuple


def get_con_num_cha_per_con_num(m, m_is_concept_num, total_cha):
    if m_is_concept_num:
        concept_num = m
        cha_per_con = total_cha // m
    else:
        concept_num = total_cha // m
        cha_per_con = m
    return concept_num, cha_per_con


def load_concept(concept_covs, concept_means, eigen_topk=1, concept_mode="pca",
                 direction_by_value=False, max_resp_value=None, min_resp_value=None) -> Tuple[list, list]:
    concept_vecs = []
    concept_means_norm = []
    for layer_i in range(len(concept_means)):
        concept_means_norm.append(concept_means[layer_i] / (torch.norm(concept_means[layer_i], dim=1, p=2, keepdim=True) + 1e-16))

    if concept_mode.lower() == "pca":
        for layer_i in range(len(concept_covs)):
            concept_vec = torch.linalg.eigh(concept_covs[layer_i])[1][:, :, -eigen_topk]
            concept_vec_norm = concept_vec / (torch.norm(concept_vec, dim=1, keepdim=True, p=2) + 1e-16)
            mask = torch.sum(concept_means_norm[layer_i] * concept_vec_norm, dim=1)
            mask = torch.where(mask > 0, 1., -1.)
            concept_vec = concept_vec * mask[:, None]
            concept_vec = concept_vec / (torch.norm(concept_vec, dim=1, keepdim=True, p=2) + 1e-16)
            concept_vecs.append(concept_vec)
    elif concept_mode.lower() == "sevec":
        cur_concept = concept_means_norm[0][..., 0]
        for layer_i in range(1, len(concept_means_norm)):
            next_concept = concept_means_norm[layer_i][..., 0]
            concept_vecs.append(cur_concept)
            cur_concept = next_concept
        concept_vecs.append(cur_concept)
    else:
        assert False, "No exists concept method!"

    if direction_by_value:
        for layer_i in range(len(concept_vecs)):
            for concept_i in range(concept_vecs[layer_i].shape[0]):
                if abs(max_resp_value[layer_i][concept_i][0]) < abs(min_resp_value[layer_i][concept_i][0]):
                    concept_vecs[layer_i][concept_i] = -concept_vecs[layer_i][concept_i]

    return concept_vecs, concept_means


def cal_cov_component(features, Sum_A, Square_Sum_A, cov_xx, cov_mean, args):
    for layer_i, feat in enumerate(features):
        if len(feat.shape) == 4:
            B, C, H, W = feat.shape
            feat = feat.reshape(B, args.concept_per_layer[layer_i], -1, H, W).permute(1, 2, 0, 3, 4)
        elif len(feat.shape) == 3:
            B, C, N = feat.shape
            feat = feat.reshape(B, args.concept_per_layer[layer_i], -1, N).permute(1, 2, 0, 3)

        feat = torch.flatten(feat, 2)
        strength = torch.norm(feat, p=2, dim=1, keepdim=True)
        ori_feat = feat
        feat = feat * strength
        Sum_A[layer_i] += torch.sum(strength.squeeze(1), dim=1)
        Square_Sum_A[layer_i] += torch.sum(strength.squeeze(1) ** 2, dim=1)
        cov_xx[layer_i] += torch.bmm(feat, ori_feat.permute(0, 2, 1))
        cov_mean[layer_i] += torch.sum(feat, dim=-1, keepdim=True)
    return Sum_A, Square_Sum_A, cov_xx, cov_mean


def cal_cov(cov_xx, cov_mean, Sum_A):
    cov_xx /= Sum_A[:, None, None]
    cov_mean /= Sum_A[:, None, None]
    cov = cov_xx - torch.bmm(cov_mean, cov_mean.permute(0, 2, 1))
    return cov, cov_mean[..., 0]


def cal_concept(cov, cov_mean):
    evalue, evector = torch.linalg.eigh(cov)
    cov_mean_norm = cov_mean / (torch.norm(cov_mean, dim=1, p=2, keepdim=True) + 1e-16)
    evector[:, :, -1] /= (torch.norm(evector[:, :, -1], dim=1, p=2, keepdim=True) + 1e-16)
    mask = torch.where(torch.sum(evector[:, :, -1] * cov_mean_norm, dim=1) > 0, 1, -1)
    concept_vector = evector[:, :, -1] * mask[:, None]
    concept_vector /= (torch.norm(concept_vector, dim=1, p=2, keepdim=True) + 1e-16)
    concept_vector = concept_vector.type(torch.float32)
    concept_mean = cov_mean.type(torch.float32)
    return concept_vector, concept_mean
