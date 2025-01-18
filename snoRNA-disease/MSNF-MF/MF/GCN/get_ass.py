import numpy as np
import pandas as pd

df = pd.read_excel(r'D:\mycode\snoRNA-disease\data\adj.xlsx', index_col=0)
print(df.shape)


def write_snoRNA_disease_relationships(df, output_file):
    with open(output_file, 'w') as f:
        for miRNA, row in df.iterrows():
            for stress, value in row.items():
                if value == 1:
                    f.write(f"{miRNA}\t{stress}\n")


write_snoRNA_disease_relationships(df, 'ass.txt')
