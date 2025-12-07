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

# Specify which pickle file to load
input_file = 'AB_decay_weighted.pickle'  # Use 'AB_decay.pickle' for original unweighted version
with open(input_file, 'rb') as f:
	fid_AB = pk.load(f)
	endpoints_AB = fid_AB["endpoints"]
	fid_AB_means = fid_AB["decay data"][0]
	fid_AB_data = fid_AB["decay data"][1]
	# Load alpha from pickle if available, otherwise use default value
	alpha = fid_AB.get("alpha", 0.95)


#Compute exponential fit

# Prepare data for fitting
m_values = np.array(range(endpoints_AB[0], endpoints_AB[1]+1))
fidelity_values = np.array([fid_AB_means[i] for i in range(endpoints_AB[0], endpoints_AB[1]+1)])

# Weighted least squares fitting
# Choose one of the following methods:

# --- Method A: Theoretical weights (proportional to sqrt(n_samples)) ---
# Assumes equal intrinsic noise at all bounce numbers
# If samples_per_bounce is not available in pickle, assume uniform sampling
samples_per_bounce = fid_AB.get("samples_per_bounce", None)
if samples_per_bounce is None:
    # Uniform sampling: use unweighted fit
    popt_AB, pcov_AB = curve_fit(exp, m_values, fidelity_values)
else:
    # Method A: weights = sqrt(n_samples) because variance ∝ 1/n_samples
    # Uncomment the following 2 lines to use Method A:
    # weights_A = np.sqrt([samples_per_bounce[m] for m in m_values])
    # popt_AB, pcov_AB = curve_fit(exp, m_values, fidelity_values, sigma=1/weights_A, absolute_sigma=False)

    # Method B: Empirical weights (based on actual sample standard deviation)
    # Uses the actual variability observed in the data
    std_errors = []
    for m in m_values:
        fidelity_samples = fid_AB_data[m]
        sample_std = np.std(fidelity_samples, ddof=1)  # Sample standard deviation
        n_samples = samples_per_bounce.get(m, len(fidelity_samples))
        std_errors.append(sample_std / np.sqrt(n_samples))  # Standard error of the mean

    # Comment out the following 2 lines to use Method A instead:
    weights_B = np.array(std_errors)
    popt_AB, pcov_AB = curve_fit(exp, m_values, fidelity_values, sigma=weights_B, absolute_sigma=True)


#Compute studentized confidence interval correction
h = t.ppf((1 +0.95) / 2., 18-2)


#Set up figure
g = plt.figure(2)
ax2 = plt.subplot()


#Plot sequence length averages
ax2.scatter(range(endpoints_AB[0],endpoints_AB[1]+1), [fid_AB_means[i] for i in range(endpoints_AB[0],endpoints_AB[1]+1)],color = "b",label = r"$\alpha =0.97$")

#Plot individual sequence means
# Handle variable number of samples per bounce
max_samples = max(len(fid_AB_data[i]) for i in range(endpoints_AB[0], endpoints_AB[1]+1))
for k in range(max_samples):
	bounce_nums = []
	fidelities = []
	for i in range(endpoints_AB[0], endpoints_AB[1]+1):
		if k < len(fid_AB_data[i]):  # Check if this bounce has this sample
			bounce_nums.append(i)
			fidelities.append(fid_AB_data[i][k])
	if bounce_nums:  # Only plot if we have data
		ax2.scatter(bounce_nums, fidelities, color="b", alpha=0.2, s=5)

#Add network link fidelity
ab_fid  = f"Network link fidelity = {popt_AB[1]:.3f}" f" $\\pm${h*np.sqrt(pcov_AB[1,1]):.3f}"
ax2.text(5.5,0.45,ab_fid,fontsize = 15)
#Plot exponential decay
ax2.plot(range(endpoints_AB[0],endpoints_AB[1]+1), [exp(m, popt_AB[0],popt_AB[1]) for m in range(endpoints_AB[0],endpoints_AB[1]+1)] ,color = "b",alpha = 0.7)


#set axes labels
ax2.set_xlabel("Number of A $\\to$ B $\\to$ A bounces",fontsize =18)
ax2.set_ylabel("Sequence mean $b_m$",fontsize =18)
ax2.set_xticks(np.arange(2,21,2))
ax2.legend(loc='best',fontsize=12)


#Save figure
g.savefig("two_node_netrb.pdf",transparent=True)
print("Figure saved as two_node_netrb.pdf")