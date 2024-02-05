import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
x = list(range(0, 101, 5))
data = [1.0, 0.8443708609271523, 0.6225165562913907, 0.4337748344370861, 0.3344370860927152, 0.26158940397350994,
        0.2185430463576159, 0.18543046357615894, 0.16225165562913907, 0.1357615894039735, 0.12582781456953643,
        0.11258278145695365, 0.10927152317880795, 0.10264900662251655, 0.09933774834437085, 0.09933774834437085,
        0.08609271523178808, 0.0728476821192053, 0.059602649006622516, 0.056291390728476824, 0.056291390728476824]
data = [i * 100 for i in data]
acc = ax.bar(x, data, width=5, color='#7386c9', edgecolor='#4666f9', label='acc')
lns = acc
labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=0, fontsize=17)

ax.grid(linestyle='dashed', alpha=0.5)
ax.set_xlabel('Cumulative Negative Feedback Count', fontsize=15)
ax.set_ylabel('Retention Rate(%)', fontsize=15)
ax.set_ylim((0, 105))
ax.set_xlim((-5, 105))
# xlabels = ['None', 'N1', 'N2', 'E', 'E+N1', 'E+N2', 'E+N1+N2']
ax.tick_params(axis='y', labelsize=15)
ax.tick_params(axis='x', labelsize=15)
plt.subplots_adjust(wspace=0)
# ax.set_xticks(x, xlabels, fontsize=12)
plt.savefig("1.pdf", bbox_inches='tight')
