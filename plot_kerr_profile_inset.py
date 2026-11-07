"""Regenerate the magnified spectral diagnostic (Figure 6).

Uses the identical synthetic observation and seed as kerr_teukolsky_study.py.
The denser local candidate grid is diagnostic only; no retraining or CI claim.
Run: python plot_kerr_profile_inset.py
Requires: numpy scipy torch matplotlib, and kerr_teukolsky_study.py in same folder.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from kerr_teukolsky_study import reference, LO, HI

omega, a_true, sigma, seed = 0.5, 0.30, 0.01, 0
x = np.linspace(-0.9, 0.9, 81)
rng = np.random.default_rng(seed + int(a_true * 1000) + int(sigma * 10000))
y = reference(a_true, x, omega, 22) + rng.normal(0, sigma, len(x))
coarse = np.linspace(LO, HI, 100)
fine = np.linspace(0.278, 0.322, 1401)
sums_coarse = np.array([np.sum((reference(float(a),x,omega,22)-y)**2) for a in coarse])
sums_fine = np.array([np.sum((reference(float(a),x,omega,22)-y)**2) for a in fine])
minimum = min(sums_coarse.min(), sums_fine.min())
full_dev = (sums_coarse-minimum)/sigma**2
local_dev = (sums_fine-minimum)/sigma**2
fig,ax = plt.subplots(figsize=(8.2,5.5))
ax.plot(coarse,full_dev,color='#2478a2',lw=1.6,label='100-point full-range scan')
ax.axvline(a_true,color='#b34742',ls=':',lw=1.3,label='Generating spin (0.30)')
ax.set(xlabel='Candidate spin $a$',ylabel=r'$\Delta\chi^2$ (relative sum of squares)',title='Angular spectral-profile diagnostic')
ax.set_xlim(0,1)
ax.set_ylim(bottom=0)
inset=inset_axes(ax,width='55%',height='56%',loc='upper right',borderpad=1.6)
inset.plot(fine,local_dev,color='#2478a2',lw=1.7)
inset.axhline(3.841459,color='#ba8730',ls='--',lw=1.3)
inset.axvline(a_true,color='#b34742',ls=':',lw=1.2)
inset.set_xlim(.286,.314)
inset.set_ylim(0,32)
inset.tick_params(labelsize=8)
inset.set_title('Magnification: same observations',fontsize=9)
inset.text(.287,4.6,r'Nominal $3.84$ cutoff',fontsize=7.5,color='#91651b')
ax.legend(loc='upper left',fontsize=9)
fig.subplots_adjust(left=.105,right=.97,bottom=.13,top=.91)
fig.savefig(Path(__file__).resolve().parent/'kerr_fig_profile.pdf',bbox_inches='tight')
plt.close(fig)
print('Figure 6 generated; local grid min:',round(float(fine[sums_fine.argmin()]),6))
print('The cutoff is not a coverage-validated confidence interval.')
