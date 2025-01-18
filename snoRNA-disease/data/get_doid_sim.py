import pandas as pd
import numpy as np
import obonet
import networkx as nx
import math

# 通过obonet库读取疾病本体数据
# url = "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo"
locals_path = "doid.obo"
HDO_net = obonet.read_obo(locals_path)


def get_SV(disease, w):
    S = HDO_net.subgraph(nx.descendants(HDO_net, disease) | {disease})
    SV = dict()
    shortest_paths = nx.shortest_path(S, source=disease)
    for x in shortest_paths:
        SV[x] = math.pow(w, (len(shortest_paths[x]) - 1))
    return SV


def get_similarity(d1, d2, w):
    SV1 = get_SV(d1, w)
    SV2 = get_SV(d2, w)
    intersection_value = 0
    for disease in set(SV1.keys()) & set(SV2.keys()):
        intersection_value = intersection_value + SV1[disease]
        intersection_value = intersection_value + SV2[disease]
    return intersection_value / (sum(SV1.values()) + sum(SV2.values()))


def getDisNet(dilen, disease, w):
    diSiNet = np.zeros((dilen, dilen))
    for d1 in range(dilen):
        if disease[d1] in HDO_net.nodes:
            for d2 in range(d1 + 1, dilen):
                if disease[d2] in HDO_net.nodes:
                    diSiNet[d1, d2] = diSiNet[d2, d1] = get_similarity(
                        disease[d1], disease[d2], w
                    )
    return diSiNet


data = pd.read_excel(r'disease_doid.xlsx')
d_names = list(data['Disease Name'])
do_ids = list(data['DO ID'])
s_d1 = getDisNet(len(do_ids), do_ids, 0.5)
np.fill_diagonal(s_d1, 1)
# s_d1 = normalize_matrix(s_d1)
print(s_d1.shape)
np.savetxt(r"..\data\Similarity matrix\doid_sim.txt", s_d1)