import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent



def read_exp_data_compr(filename,rowsskipped=3):
    """_summary_

    Args:
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    data = np.loadtxt(filename,delimiter="\t",skiprows=rowsskipped,
        converters=lambda x: float(x.replace(",", ".")))
    L=data[:,1]
    F=data[:,0]
    #T=data[:,2]
    
    l_min_pos = np.where(L>=5.0)[0][0]
    l_max_pos = np.where(L==max(L))[0][0]

    #print(l_min_pos)
    #print(l_max_pos)

    Lcut = L[l_min_pos:l_max_pos]
    Fcut = F[l_min_pos:l_max_pos]

    #Tcut = T[l_min_pos:l_max_pos]

    
    Lcut -= Lcut[0]
    Lcut /= 1000#mm to m
    
    Fcut -= Fcut[0]
    #Tcut -= Tcut[0]

    Lcut = -Lcut
    Fcut = -Fcut

    #return np.column_stack((Lcut, Fcut, Tcut))
    return (Lcut,Fcut)

def read_exp_data_tens(filename,rowsskipped=3):
    """_summary_

    Args:
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    data = np.loadtxt(filename,delimiter="\t",skiprows=rowsskipped,
            converters=lambda x: float(x.replace(",", ".")))
    L=data[:,1]
    F=data[:,0]
    #T=data[:,2]
    
    L = -L
    #plot_data((L,F),exp_name="l,f",xlabel="L,mm",ylabel="F,N")

    
    l_max_pos = np.where(L==max(L))[0][0]

    Lcut = L[0:l_max_pos]
    Fcut = F[0:l_max_pos]
    #Tcut = T[0:l_max_pos]
    
    Lcut -= Lcut[0]
    Lcut /= 1000#mm to m
    Fcut -= Fcut[0]

    return (Lcut,Fcut)

def read_cut_data(filename):
    """_summary_
    """
    data = np.loadtxt(filename,delimiter="\t",skiprows=1)
    L=data[:,0]
    F=data[:,1]
    
    return(L,F)

_fig,ax1 = plt.subplots()

_fig,ax2 = plt.subplots()

_fig,ax3 = plt.subplots()



c1 = read_exp_data_compr(BASE_DIR /"txtfiles" / "COMPR1.tab")
c2 = read_exp_data_compr(BASE_DIR /"txtfiles" / "COMPR2.tab")
c3 = read_exp_data_compr(BASE_DIR /"txtfiles" / "COMPR3.tab")
c4 = read_exp_data_compr(BASE_DIR /"txtfiles" / "COMPR4.tab")
c5 = read_exp_data_compr(BASE_DIR /"txtfiles" / "COMPR5.tab")


cavg = ((c1[0]+c2[0]+c3[0]+c4[0]+c5[0])/5,(c1[1]+c2[1]+c3[1]+c4[1]+c5[1])/5)
cavg = np.array(cavg)

cs = [c1,c2,c3,c4,c5]

for i in range(len(cs)):
    ax1.plot(cs[i][0],cs[i][1],label=i,linewidth=2)

ax1.plot(cavg[0],cavg[1],label="avg",linewidth=4)


np.savetxt(BASE_DIR /"txtfiles" / "COMPRAVG.tab", cavg.T, fmt="%.6f", delimiter="\t",header="Lambda\tSigma")



t1 = read_exp_data_tens(BASE_DIR /"txtfiles" / "TENS1.tab")
t2 = read_exp_data_tens(BASE_DIR /"txtfiles" / "TENS2.tab")
t3 = read_exp_data_tens(BASE_DIR /"txtfiles" / "TENS3.tab")
t4 = read_exp_data_tens(BASE_DIR /"txtfiles" / "TENS4.tab")
t5 = read_exp_data_tens(BASE_DIR /"txtfiles" / "TENS5.tab")
t6 = read_exp_data_tens(BASE_DIR /"txtfiles" / "TENS6.tab")

ts = [t1,t2,t3,t4,t5,t6]

for i in range(len(ts)):
    ax2.plot(ts[i][0],ts[i][1],label=i,linewidth=2)

tavg = ((t1[0]+t2[0]+t3[0]+t4[0]+t5[0]+t6[0])/6,(t1[1]+t2[1]+t3[1]+t4[1]+t5[1]+t6[1])/6)
tavg = np.array(tavg)

ax2.plot(tavg[0],tavg[1],label="avg",linewidth=4)


np.savetxt(BASE_DIR /"txtfiles" / "TENSAVG.tab", tavg.T, fmt="%.6f", delimiter="\t",header="L\tF")


c = read_cut_data(BASE_DIR /"txtfiles" / "COMPRAVG.tab")
t = read_cut_data(BASE_DIR /"txtfiles" / "TENSAVG.tab")

ax3.plot(c[0],c[1],label="comp_avg")
ax3.plot(t[0],t[1],label="tens_avg")
ax1.legend()
ax2.legend()
ax3.legend()
plt.show()