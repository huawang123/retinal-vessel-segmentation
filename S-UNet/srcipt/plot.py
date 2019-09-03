from matplotlib import pyplot as plt

UNet = ['S-Unet', 'red', 0.9821, 0.8303, 0.21]
DRIU = ['DRIU', 'green', 0.9793, 0.8222, 7.5]
# ERFNet = ['ERFNet', 'blue', 0.9, 0.8022, 2.06]
M2U = ['SWT-UNet', 'orange', 0.9821, 0.8281, 0.7]
UNet1 = ['UNet', 'purple', 0.9755, 0.8142, 35]
bts1 = ['BTS-Net', 'chocolate', 0.9796, 0.8208, 7.87]
munet = ['Mi-UNet', 'blue', 0.9799, 0.8231, 0.07]

nets = [UNet, DRIU, M2U, UNet1, bts1, munet]

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for net in nets:
    name, color, madd, dice, params = net
    ax.scatter(x=[madd], y=[dice], s=[params * 100],  c=[color], marker='o', label=name)
    # ax.scatter(x=[madd], y=[dice], c=[color], marker='o', label=name)
    # ax.scatter(x=[madd], y=[dice], c='black', marker='.')

ax.set_xlabel('AUC')
ax.set_ylabel('F1-scores')
ax.set_xlim(left=0.97, right=0.985)
ax.set_ylim(bottom=0.805,top=0.833)
# ax.set_ylim(top=0.84)

# ax.legend()
ax.grid()

fig.savefig('plotttt.png', dpi=300)
plt.close(fig)