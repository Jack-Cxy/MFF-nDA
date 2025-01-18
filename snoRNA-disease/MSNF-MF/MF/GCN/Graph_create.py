import dgl
import torch


def get_ass(file_path):
    piRNA_name = []
    disease_name = []
    piRNA_disease = []

    f = open(file_path, 'r', encoding='utf-8')
    contents = f.readlines()
    for content in contents:
        value = content.split('\t')
        value[0] = value[0].lower()
        if value[0] not in piRNA_name: piRNA_name.append(value[0])
        value[1] = value[1].strip('\n')
        if value[1] not in disease_name: disease_name.append(value[1])
        piRNA_disease.append(value)
    f.close()

    piRNA_num = len(piRNA_name)
    disease_num = len(disease_name)

    print(piRNA_num)
    print(disease_num)

    piRNA_index = dict(zip(piRNA_name, range(0, piRNA_num)))
    disease_index = dict(zip(disease_name, range(0, disease_num)))

    input_piRNA_disease = [[], []]
    for i in range(len(piRNA_disease)):
        input_piRNA_disease[0].append(piRNA_index.get(piRNA_disease[i][0]))
        input_piRNA_disease[1].append(disease_index.get(piRNA_disease[i][1]))

    print(len(input_piRNA_disease[0]))

    return input_piRNA_disease


def Graph_create(snoRNA_disease, snoRNA_feat, disease_feat):
    graph = {
        ('snoRNA', 'p_d', 'disease'): (torch.tensor(snoRNA_disease[0]), torch.tensor(snoRNA_disease[1])),
        ('disease', 'd_p', 'snoRNA'): (torch.tensor(snoRNA_disease[1]), torch.tensor(snoRNA_disease[0])),
    }

    graph = dgl.heterograph(graph)

    graph.nodes['snoRNA'].data['feature'] = snoRNA_feat
    graph.nodes['disease'].data['feature'] = disease_feat

    graph_h = {'snoRNA': graph.nodes['snoRNA'].data['feature'],
               'disease': graph.nodes['disease'].data['feature']}

    return graph, graph_h

