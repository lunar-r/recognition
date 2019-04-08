# 快速修改文件名

import os

path = "Donald_Trump/"
f = os.listdir(path)
n = 0
for item in f:
    old = path + f[n]
    new = path + "Donald_Trump_" + str(n + 1000) + ".jpg"
    os.rename(old, new)
    print(old, " --> ", new)
    n = n + 1

