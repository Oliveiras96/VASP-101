import os
import numpy as np
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

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

# cm = plt.cm.get_cmap('cividis')
cm = mpl.colormaps['coolwarm']

###############################################################################################################################################    
def bndplot(bandsfile,subplot):
    data = np.loadtxt(bandsfile)

    # Get the kpoints (x coordinates):
    kpoints = data[:,0]
    # Get the bands eigenvalues for each k-point:
    bands = data[:, 1:]

    # Plot the bands:
    for i in range(bands.shape[1]):
        subplot.plot(kpoints, 
                     bands[:, i], 
                     lw=1.25,
                     alpha=0.35,
                     color='k')


def bndplot_proj(bandsfile,subplot,ylims, symmetry_points, symmetry_labels, title):
    # The REFORMAED_BAND.dat vaspkit output file has the following format: each line has the kpoints (first column)
    # followed by the n-eigenvalues for that kpoint.

    # Read the REFORMATED_BAND.dat file:
    data = np.loadtxt(bandsfile)

    kpoints = np.unique(data[:,0])
    bands = []
    
    for kpoint in kpoints:
        bands.append([[x[1], x[11]] for x in data if x[0] == kpoint])

    # print("Len(kpoints): ", len(np.unique(data[:,0])))
    # print("Len(band[0]): ", len(bands[0]))
    
    for kpoint, band in zip(kpoints, bands):
        y_values = [x[0] for x in band]
        y_weights = [x[1] for x in band]

        ax = subplot.scatter([kpoint]*len(band), y_values, c=y_weights, cmap=cm, marker='.', s=7.5)
        # ax = subplot.scatter([kpoint]*len(band), y_values, marker='.')
        # subplot.colorbar(ax, values=y_weights)
        # map = axs_.imshow(np.stack([[kpoint]*len(band), y_values]), c=y_weights, cmap='cividis')

    
    subplot.set_ylim([ylims[0], ylims[1]])
    subplot.set_xticks([])
    subplot.set_xlim([min(kpoints), max(kpoints)])

    # Plot Fermi Level:
    subplot.axhline(0, 0, 1, ls='--', color='r')

    # Title:
    # subplot.set_title(title, fontsize=15)

    # Plot high-symmetry labels:
    for i, j in zip(symmetry_labels, symmetry_points):
        subplot.axvline(j, 0, 1, linestyle='--', color='k', lw=1.0)
        subplot.text(j-0.01*(max(kpoints)-min(kpoints)), ylims[0]-(0.075*abs(ylims[1]-ylims[0])), i, fontsize=15)
    
    return ax
###############################################################################################################################################    

# fig, axs = plt.subplots(1, 2, figsize=(12,6))
fig, axs = plt.subplots()

ylims = [-2,2]

high_symmetry_labels = [r'$\Gamma$', 'M', 'K', r'$\Gamma$']
# high_symmetry_points = [0.000, 0.486, 0.767, 1.328]
high_symmetry_points = [0.000, 1.139, 1.797, 3.112]

linestyle = '-'
linecolor = 'k'

bandsfile_path  = "./NSCF-new/PBAND_Mo_UP.dat"

# bndplot(bandsfile="Bi2Te3/VASP/NSCF-new/REFORMATTED_BAND.dat", subplot=axs)
bndplot(bandsfile="./NSCF-new/REFORMATTED_BAND_UP.dat", subplot=axs)

# bandsfile_path = "Bi2Te3/VASP/NSCF-new/PBAND_Bi.dat"

ax = bndplot_proj(bandsfile=bandsfile_path,
        subplot=axs,
        ylims=ylims,
        symmetry_labels=high_symmetry_labels,
        symmetry_points=high_symmetry_points,
        title=r"")

axs.set_ylabel(r"E - $E_F$ (eV)", fontsize=12)

# cbar = fig.colorbar(ax,
#              ticks=[0,1])
# cbar.ax.set_yticklabels(['S', 'Fe'])

# ax = bndplot(bandsfile="FeS2-pirite/VASP/electronic-structure+U/NSCF/PBAND_S.dat",
#         subplot=axs[1],
#         ylims=ylims,
#         symmetry_labels=high_sym_labels,
#         symmetry_points=high_sym_points,
#         title=r"$Te_{p}$")

cmap = plt.colorbar(ax)
cmap.set_ticks([])

plt.tight_layout()
# plt.savefig("Bi2Te3-projected-bands-new-Feb3rd.png", dpi=300)
plt.savefig("MoS2-projected-bands.png", dpi=300)
plt.show()