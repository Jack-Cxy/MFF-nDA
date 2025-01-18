import pandas as pd
import numpy as np
from utils import seed_everything
seed_everything(42)


trans_feat = np.load(r'D:\mycode\snoRNA-disease\data\feat\trans_feat.npy')
print(trans_feat.shape)
gat_feat = np.load(r'D:\mycode\snoRNA-disease\data\feat\gat_feat.npy')
print(gat_feat.shape)
gcn_feat = np.load(r'D:\mycode\snoRNA-disease\data\feat\gcn_feat.npy')
print(gcn_feat.shape)
label = pd.read_csv(r'D:\mycode\snoRNA-disease\MSNF-MF\MF\Transformer\label_bal.csv',index_col=0)
index = label.index
print(label.shape)

feat = (trans_feat + gat_feat + gcn_feat) / 3.0
print(feat.shape)

f = pd.DataFrame(feat)
f.index = index
f1 = pd.concat([f,label.iloc[:,-1]],axis=1)
print(f1.shape)
f1.sample(frac=1).to_csv(r'D:\mycode\snoRNA-disease\data\result\feat_final_all.csv')


