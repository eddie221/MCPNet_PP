import torch


def KL_div(x, y):
    return torch.sum(x * (torch.log2(x) - torch.log2(y)), dim=-1)


def JS_div(x, y):
    M = (x + y) / 2
    return (KL_div(x, M) + KL_div(y, M)) / 2


def cal_JS_sim(dist1, dist2):
    assert dist1.shape[-1] == dist2.shape[-1], \
        f"Error shape of dist1 {dist1.shape} and dist2 {dist2.shape}!"
    return JS_div(dist1 + 1e-8, dist2 + 1e-8)


def l2_dist(x, y):
    return torch.norm(x - y, p=2, dim=-1)


def cal_l2_sim(dist1, dist2):
    assert dist1.shape[-1] == dist2.shape[-1], \
        f"Error shape of dist1 {dist1.shape} and dist2 {dist2.shape}!"
    return l2_dist(dist1, dist2)


def cal_sim(img_MCP_dist, class_MCP_dist, dist_func="JS"):
    if dist_func == "JS":
        return cal_JS_sim(img_MCP_dist, class_MCP_dist)
    elif dist_func == "l2":
        return cal_l2_sim(img_MCP_dist, class_MCP_dist.unsqueeze(0))
    else:
        assert False, f"Not implement distance function : {dist_func}"


def cal_acc(feats, cent_tree_nodes, concept_vecs, concept_means, args):
    max_responses = []
    for layer_i, feat in enumerate(feats):
        if len(feat.shape) == 4:
            B, C, H, W = feat.shape
            feat = feat.reshape(B, C // args.concept_cha[layer_i], args.concept_cha[layer_i], H, W)
            feat = feat - concept_means[layer_i].unsqueeze(0).unsqueeze(3).unsqueeze(4)
            feat_norm = feat / (torch.norm(feat, dim=2, keepdim=True) + 1e-16)
            concept_vector = concept_vecs[layer_i].cuda(args.global_rank)
            response = torch.sum(feat_norm * concept_vector.unsqueeze(0).unsqueeze(3).unsqueeze(4), dim=2)
            max_response = torch.nn.functional.adaptive_max_pool2d(response, output_size=1)
            max_response = max_response.squeeze(-1).squeeze(-1)
        elif len(feat.shape) == 3:
            B, C, N = feat.shape
            feat = feat.reshape(B, C // args.concept_cha[layer_i], args.concept_cha[layer_i], N)
            feat = feat - concept_means[layer_i].unsqueeze(0).unsqueeze(3)
            feat_norm = feat / (torch.norm(feat, dim=2, keepdim=True) + 1e-16)
            concept_vector = concept_vecs[layer_i].cuda(args.global_rank)
            response = torch.sum(feat_norm * concept_vector.unsqueeze(0).unsqueeze(3), dim=2)
            max_response = torch.nn.functional.adaptive_max_pool1d(response, output_size=1)
            max_response = max_response.squeeze(-1)
        max_responses.append(torch.clip((max_response + 1) / 2, min=1e-8, max=1))

    max_responses = torch.cat(max_responses, dim=1)
    if args.CCD_norm == "normal":
        max_responses = max_responses / torch.sum(max_responses, dim=1, keepdim=True)
    elif args.CCD_norm == "softmax":
        max_responses = torch.nn.functional.softmax(max_responses / args.MCP_temperature, dim=1)

    Diff_centroid_dist_resp = []
    for class_i in range(args.category):
        resp_sim = cal_sim(max_responses, cent_tree_nodes[class_i])
        Diff_centroid_dist_resp.append(resp_sim)

    Diff_centroid_dist_resp = torch.stack(Diff_centroid_dist_resp, dim=1)
    return torch.topk(-Diff_centroid_dist_resp, dim=1, k=1)[1], \
           torch.topk(-Diff_centroid_dist_resp, dim=1, k=5)[1]
