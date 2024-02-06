from turtle import color
from matplotlib import rcParams
import matplotlib.pyplot as plt
import re

logFile = r'test_nyu/test.log' 
# logFile = r'test_fe.log' 
text = ''
file = open(logFile)
max_auc3 = 0
for line in file:
    if("array" in line):
        res = line.split(",")
        print(res)
        auc3 = res[-3]
        print(auc3)
        max_auc3 = max(max_auc3,float(auc3))
print(max_auc3)
        # auc3 = auc.split(",")[1]
        
        # print(auc3)
        # auc_res = aucs.split(",")
        # print(auc_res)
        