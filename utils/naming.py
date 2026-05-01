def id2name(idx, concept_per_layer):
    idx += 1
    layer_i = 1
    while idx > concept_per_layer[layer_i - 1]:
        idx -= concept_per_layer[layer_i - 1]
        layer_i += 1
    return f"l{layer_i}_{idx}"


def name2id(idx, concept_per_layer):
    layer = int(idx.split("_")[0][1:])
    i_th = int(idx.split("_")[1])
    count = sum(concept_per_layer[:layer - 1])
    return count + i_th - 1
