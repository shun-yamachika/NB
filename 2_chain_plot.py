import pickle as pk
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非対話モードに設定
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t

#Plotting script for figure a

def exp(m,A,f):
	return A* f**m


plt.close()

with open('AB_decay.pickle', 'rb') as f:
	fid_AB = pk.load(f)
	endpoints_AB = fid_AB["endpoints"]
	fid_AB_means = fid_AB["decay data"][0] 
	fid_AB_data = fid_AB["decay data"][1]


#Compute exponential fit
popt_AB,pcov_AB = curve_fit(exp,np.array(range(endpoints_AB[0],endpoints_AB[1]+1)), [fid_AB_means[i] for i in range(endpoints_AB[0],endpoints_AB[1]+1)])


#Compute studentized confidence interval correction
h = t.ppf((1 +0.95) / 2., 18-2)


#Set up figure
g = plt.figure(2)
ax2 = plt.subplot()


#Plot sequence length averages
ax2.scatter(range(endpoints_AB[0],endpoints_AB[1]+1), [fid_AB_means[i] for i in range(endpoints_AB[0],endpoints_AB[1]+1)],color = "b",label = r"$\alpha =0.97$")

#Plot individual sequence means
for k in range(len(fid_AB_data[endpoints_AB[0]])):
		ax2.scatter(range(endpoints_AB[0],endpoints_AB[1]+1), [fid_AB_data[i][k] for i in range(endpoints_AB[0],endpoints_AB[1]+1)],color = "b",alpha = 0.2,s=5)

#Add network link fidelity
ab_fid  = f"Network link fidelity = {popt_AB[1]:.3f}" f" $\\pm${h*np.sqrt(pcov_AB[1,1]):.3f}"
ax2.text(5.5,0.45,ab_fid,fontsize = 15)
#Plot exponential decay
ax2.plot(range(endpoints_AB[0],endpoints_AB[1]+1), [exp(m, popt_AB[0],popt_AB[1]) for m in range(endpoints_AB[0],endpoints_AB[1]+1)] ,color = "b",alpha = 0.7)


#set axes labels
ax2.set_xlabel("Number of A $\\to$ B $\\to$ A bounces",fontsize =18)
ax2.set_ylabel("Sequence mean $b_m$",fontsize =18)
ax2.set_xticks(np.arange(2,21,2))


#Save figure
g.savefig("two_node_netrb.pdf",transparent=True)
print("Figure saved as two_node_netrb.pdf")