import numpy as np
import pandas as pd
import math


# 功能相似性
def S_fun1(DDsim, T0, T1):
    DDsim = np.array(DDsim)
    T0_T1 = []
    if len(T0) != 0 and len(T1) != 0:
        for ti in T0:
            m_ax = []
            for tj in T1:
                m_ax.append(DDsim[ti][tj])
            T0_T1.append(max(m_ax))
    if len(T0) == 0 or len(T1) == 0:
        T0_T1.append(0)
    T1_T0 = []
    if len(T0) != 0 and len(T1) != 0:
        for tj in T1:
            m_ax = []
            for ti in T0:
                m_ax.append(DDsim[tj][ti])
            T1_T0.append(max(m_ax))
    if len(T0) == 0 or len(T1) == 0:
        T1_T0.append(0)
    return T0_T1, T1_T0


def FS_fun1(T0_T1, T1_T0, T0, T1):
    a = len(T1)
    b = len(T0)
    S1 = sum(T0_T1)
    S2 = sum(T1_T0)
    FS = []
    if a != 0 and b != 0:
        Fsim = (S1 + S2) / (a + b)
        FS.append(Fsim)
    if a == 0 or b == 0:
        FS.append(0)
    return FS


SD = pd.read_excel(r"adj.xlsx", index_col=0)
print(SD.shape)

DS = np.loadtxt(r'..\data\Similarity matrix\doid_sim.txt')
print(DS.shape)

# 计算功能相似性
m = SD.shape[0]
T = []
for i in range(m):
    T.append(np.where(SD.iloc[i] == 1))

Fs = []
for ti in range(m):
    for tj in range(m):
        Ti_Tj, Tj_Ti = S_fun1(DS, T[ti][0], T[tj][0])
        FS_i_j = FS_fun1(Ti_Tj, Tj_Ti, T[ti][0], T[tj][0])
        Fs.append(FS_i_j)
Fs = np.array(Fs).reshape(SD.shape[0], SD.shape[0])
Fs = pd.DataFrame(Fs)
for index, rows in Fs.iterrows():
    for col, rows in Fs.items():
        if index == col:
            Fs.loc[index, col] = 1
print(Fs.shape)
# Fs = normalize_matrix(Fs)
np.savetxt(r"..\data\Similarity matrix\Func_sim.txt", Fs)
