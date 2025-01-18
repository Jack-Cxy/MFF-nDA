import pandas as pd
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def optimized_matrix(a, b, match_score=3, gap_cost=2):
    len_a, len_b = len(a) + 1, len(b) + 1
    H = np.zeros((len_a, len_b), dtype=np.int32)

    for i in range(1, len_a):
        for j in range(1, len_b):
            match = H[i-1, j-1] + (match_score if a[i-1] == b[j-1] else -match_score)
            delete = H[i-1, j] - gap_cost
            insert = H[i, j-1] - gap_cost
            H[i, j] = max(match, delete, insert, 0)

    return H.max()


def compute_similarity(index_seq, j_seq):
    return optimized_matrix(index_seq, j_seq)


def parallel_similarity(index_seq, j_name, piRNA_seq_dict):
    if j_name not in piRNA_seq_dict:
        print(f"Warning: {j_name} not found in piRNA_seq_dict.")
        return 0  # 或者其他处理方式
    j_seq = piRNA_seq_dict[j_name]
    return compute_similarity(index_seq, j_seq)


def main():
    # 读取数据
    piRNA_seq = pd.read_excel(r'snoRNA-seq.xlsx')
    piRNA_seq_dict = dict(piRNA_seq.values)

    # 初始化相似性矩阵
    p2p = pd.DataFrame(columns=list(piRNA_seq_dict.keys()), index=list(piRNA_seq_dict.keys()))
    start_time = time.time()

    for cnt, (index, row) in enumerate(p2p.iterrows()):
        print(f"Processing {cnt + 1} / {len(p2p)}")
        row_index2 = row.index[cnt + 1:]  # 避免自我比较
        index_seq = piRNA_seq_dict[index]

        with ProcessPoolExecutor() as executor:
            # 使用 partial 来固定 index_seq 和 piRNA_seq_dict
            results = executor.map(partial(parallel_similarity, index_seq, piRNA_seq_dict=piRNA_seq_dict), row_index2)

        for j_name, sim in zip(row_index2, results):
            len_index = len(index_seq)
            len_j_seq = len(piRNA_seq_dict[j_name])
            p2p.at[index, j_name] = sim / np.sqrt(len_index * len_j_seq) / 3

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds.")
    p2p.to_csv(r"half_s2s_smith.csv")

    half_p2p_smith = pd.read_csv(r"half_s2s_smith.csv", index_col=0)
    half_p2p_smith_values = half_p2p_smith.values

    # 替换 NaN 为 0 并使矩阵对称
    half_p2p_smith_values = np.nan_to_num(half_p2p_smith_values)  # 替换 NaN 为 0
    half_p2p_smith_values += half_p2p_smith_values.T
    np.fill_diagonal(half_p2p_smith_values, 1)

    p2p_smith = pd.DataFrame(half_p2p_smith_values, columns=half_p2p_smith.index, index=half_p2p_smith.index)
    np.savetxt(r"..\data\Similarity matrix\seq_sim.txt", p2p_smith.values)
    print("Similarity matrix saved to seq_sim.txt.")


if __name__ == "__main__":
    main()
