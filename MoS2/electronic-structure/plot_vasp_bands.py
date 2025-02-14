import os
import numpy as np
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
# from plot_bands import *
# from read_pdos import projwfc_to_dataframe

import pandas as pd

mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'

# mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True


mpl.rcParams['xtick.major.size'] = 3
mpl.rcParams['xtick.major.size'] = 3

mpl.rcParams['xtick.minor.size'] = 1.5
mpl.rcParams['xtick.minor.size'] = 1.5

mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['font.family'] = 'serif'


############################################################################################
def bndplot(bandsfile,subplot,ylims, symmetry_points, symmetry_labels, linestyle, bandlabelcolor):
    # The REFORMAED_BAND.dat vaspkit output file has the following format: each line has the kpoints (first column)
    # followed by the n-eigenvalues for that kpoint.

    # Read the REFORMATED_BAND.dat file:
    data = np.loadtxt(bandsfile)

    # Get the kpoints (x coordinates):
    kpoints = data[:,0]
    # Get the bands eigenvalues for each k-point:
    # bands = np.array([row[1:] for row in data])
    bands = data[:, 1:]

    # Plot the bands:
    for i in range(bands.shape[1]):
        subplot.plot(kpoints, 
                     bands[:, i], 
                     ls=linestyle, 
                     color=bandlabelcolor,
                     lw=1.25)
        
    subplot.set_xlim([min(kpoints),max(kpoints)])
    subplot.set_ylim(ylims)
    subplot.set_ylabel(r'E - $E_{F}$', fontsize=15)

    subplot.set_xticks([])

    # Plot Fermi Level:
    subplot.axhline(0, 0,1, ls='--', lw=1.0, color='r')

    # Plot high-symmetry labels:
    for i, j in zip(high_symmetry_labels, high_symmetry_points):
        subplot.axvline(j, 0, 1, linestyle='--', color='k', lw=1.0)
        subplot.text(j-0.01*(max(kpoints)-min(kpoints)), ylims[0]-(0.05*abs(ylims[1]-ylims[0])), i, fontsize=15)

def read_pdos(filename, cols):
    data = pd.read_table(filename, # file path for graphene
                                    sep="\s+", 
                                    engine='python',
                                    header=0, 
                                    skiprows=[0], 
                                    usecols=np.arange(0,len(cols),1), 
                                    names=cols)
    
    return data

#################################################################################################

# fig, bands = plt.subplots()
fig, (bands, dos) = plt.subplots(1,2, gridspec_kw={'width_ratios': [16, 6]}, sharey=True)


high_symmetry_labels = [r'$\Gamma$', 'M', 'K', r'$\Gamma$']
# high_symmetry_points = [0.000, 0.486, 0.767, 1.328]
high_symmetry_points = [0.000, 1.139, 1.797, 3.112]

cols = ["Energy", "s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "dx2-y2", "tot"]

bndplot(bandsfile="./NSCF-new/REFORMATTED_BAND_UP.dat",
        subplot=bands,
        ylims=[-2, 2],
        symmetry_points=high_symmetry_points,
        symmetry_labels=high_symmetry_labels,
        linestyle='-',
        bandlabelcolor='k')

# bndplot(bandsfile="./NSCF/REFORMATTED_BAND_DW.dat",
#         subplot=bands,
#         ylims=[-2, 2],
#         symmetry_points=high_symmetry_points,
#         symmetry_labels=high_symmetry_labels,
#         linestyle='--',
#         bandlabelcolor='blue')

pdos_Mo_up = read_pdos("./NSCF-new/PDOS_Mo_UP.dat", cols)
pdos_Mo_dw = read_pdos("./NSCF-new/PDOS_Mo_DW.dat", cols)
pdos_S_up = read_pdos("./NSCF-new/PDOS_S_UP.dat", cols)
pdos_S_dw = read_pdos("./NSCF-new/PDOS_S_DW.dat", cols)

dos.plot(pdos_Mo_up["tot"], pdos_Mo_up["Energy"], lw=1.0, color='b', label="Mo")
# dos.plot(pdos_Pt_dw["tot"], pdos_Pt_dw["Energy"], lw=1.0, color='b', ls='--')

dos.plot(pdos_S_up["tot"], pdos_S_up["Energy"], lw=1.0, color='g', label="S")
# dos.plot(pdos_Zn_dw["tot"], pdos_Zn_dw["Energy"], lw=1.0, color='g', ls='--')

dos.set_xticks([])
dos.set_xlim([-0,10])
dos.axhline(0, 0, 1, lw=1.0, ls='--', color='r')
dos.axvline(0, 0, 1, lw=1.0, ls='--', color='k')
dos.set_xlabel("States/eV", fontsize=15)
dos.legend(frameon=False, fontsize=10)

bands.text(0.10, 0.25, r"$E_{g}$ = 1.27, Indirect", fontsize=12)

plt.tight_layout()
plt.savefig("bands-MoS2.png", dpi=300)
plt.show()