import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

sourceid=950
photfile = '/Users/x/Desktop/e26462/e26462-nd-photom.csv'
exptfile = '/Users/x/Desktop/e26462/e26462-nd-exptime.csv'
gapfile = '/Users/x/Desktop/e26462/e26462-nd-950-30s.csv'
exptime = pd.read_csv(exptfile)
photom = pd.read_csv(photfile,index_col='id')
gap = pd.read_csv(gapfile,index_col=None)

cnt = photom.loc[sourceid][
    ["aperture_sum_{i}".format(i=i) for i in np.arange(len(exptime))]
                                        ].values
cps = cnt / exptime.expt.values.flatten()
cps_err = np.sqrt(cnt) / exptime.expt.values.flatten()
t = gap["t0"] - gap["t0"].min()+15

nplots = 100
subplotratio = 3
ncols = np.floor(np.sqrt(nplots/subplotratio))
figscale = 10

figx=subplotratio*ncols/figscale
figy=np.ceil(nplots/ncols/figscale)
print(figx,figy)

plt.figure(figsize=(figx,figy))
for i in range(1,nplots+1):
    plt.subplot(np.ceil(nplots/ncols),ncols,i)
    plt.plot(t,gap['cps'],'k-')
    plt.plot(exptime['t0']-exptime['t0'].min()+60,cps,'r-',alpha=0.5)
    plt.fill_between(t,gap['cps']-3*gap['cps_err'],
                       gap['cps']+3*gap['cps_err'],color='blue',alpha=0.5)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

plt.subplots_adjust(wspace=0,hspace=0,left=0,right=1,bottom=0,top=1)

plt.savefig('/Users/x/Downloads/test.png',dpi=250,bbox_inches='tight',transparent=True)
