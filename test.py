import sys
from controlchart import *
import numpy as np

samples = np.genfromtxt('P2P1.csv', delimiter=' ')
samples2 = np.genfromtxt('P2P2.csv', delimiter=' ')
r = RChart(samples, L=3)
s = SChart(samples, L=3)
xBar = XBarChart(samples, variation_type="std", L=3)
xBar.add_phase2_samples(samples2)
xBar.show_charts()
xBar.show_phase2_chart()
print(xBar.summary())