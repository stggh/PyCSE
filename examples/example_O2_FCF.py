import numpy as np
import cse
import scipy.constants as C
from scipy.integrate.quadrature import simps
import matplotlib.pyplot as plt

# ground state
O2X = cse.Cse('O2', VT=['potentials/X3S-1.dat'])
R = O2X.R

# upper state
O2B = cse.Cse('O2', VT=['potentials/B3S-1.dat'])

O2X.solve(800)
wfX = np.transpose(O2X.wavefunction)[0][0]   # v" = 0 wavefunction

print("  v'     FCF")
v = []; fcf = []
for e in [50145, 50832, 51497, 52139, 52756, 53347, 53910, 54443,
          54944, 55410, 55838, 56227, 56572, 56873, 57128, 57338,
          57507, 57640, 57743, 57820, 57874, 57908]:
    O2B.solve(e)
    wfB = np.transpose(O2B.wavefunction)[0][0]   # v' wavefunction

    olap = (wfB * wfX)**2
    FCF = simps(olap, R) 

    v.append(O2B.vib)
    fcf.append(FCF)
    print(" {:2d}  {:10.2e}".format(v[-1], FCF))

fcf = np.array(fcf)
cseBX = np.loadtxt("cseBX", unpack=True)
plt.plot(*cseBX, 'ro')
scale = cseBX[1].max()/fcf.max()
print("scalefactor = {:g}".format(scale))
plt.plot(v, fcf*scale, 'bo')
plt.title(r"O$_2$ FCF $B-X$")
plt.show()
