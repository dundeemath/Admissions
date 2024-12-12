import numpy as np
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import odeint

c1_default=12
c2_default=0.0
c3_default=0.0

c1=4.0
c2=4.0
c3=4.0

t1=7.0
t2=12.0
t3=18.0

T_final=96
num_days=int(T_final/24)

drug_conc_default_day=[c1_default,c2_default,c3_default,0]
drug_conc_day=[c1,c2,c3,0]
t_sort_day=[0.0,t1,t2,t3]

drug_conc_default=[]
drug_conc=[]
t_sort=[]
for i in range(num_days):
    drug_conc_default=drug_conc_default+drug_conc_default_day
    drug_conc=drug_conc+drug_conc_day

    t_sort_day_i = [x + i*24.0 for x in t_sort_day]


    t_sort=t_sort+t_sort_day_i

t_sort.append(T_final)


print(t_sort)
print(drug_conc_default)
print(drug_conc)

