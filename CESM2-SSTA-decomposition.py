import xarray as xr
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt 
import pandas as pd

####### Modify #########
exp = 'SF-GHG'
fdir_out = './CESM2_clean_dataset/'
fdir_atm = '/glade/campaign/cesm/collections/CESM2-SF/timeseries/atm/proc/tseries/month_1/'
# '/glade/campaign/cgd/cesm/CESM2-LE/timeseries/atm/proc/tseries/month_1/'
SPNA_domain_boundary = [45, 60, 310, 340] # [south_lat, north_lat, west_lon, east_lon]
########################

###### Experiments & Simulation Identifiers
# names of variables required for the analysis
vars = ['SST','LHFLX', 'SHFLX', 'FSNS', 'FSDS', 'FSDSC', 'FLNS', 'FLDS', 'FLDSC']

ens_idx = {}
ens_idx['LE']=[]
macro_idx = ['1231','1251','1281','1301']
for macro in macro_idx:
    for i in range(20):
        ens_idx['LE'].append(macro+"."+"{:0>3}".format(i+1)) 
macro_idx = list(np.arange(1001,1200,20))
for i in range(10):
    ens_idx['LE'].append(str(macro_idx[i])+"."+"{:0>3}".format(i+1))
macro_idx = list(np.arange(1011,1200,20))    
for i in range(10):
    ens_idx['LE'].append(str(macro_idx[i])+"."+"{:0>3}".format(i+1))
    
ens_idx['SF-GHG']=[]
ens_idx['SF-BMB']=[]
ens_idx['SF-EE']=[]

for j in range(15):
    ens_idx['SF-GHG'].append("{:0>3}".format(j+1))
    ens_idx['SF-BMB'].append("{:0>3}".format(j+1))
for j in range(15):
    ens_idx['SF-EE'].append("1"+"{:0>2}".format(j+1))
    
ens_idx['SF-AAER']=[]
for j in range(20):
    ens_idx['SF-AAER'].append("{:0>3}".format(j+1))
    
nyear = {}
nyear['LE'] = 251
nyear['SF-GHG'], nyear['SF-AAER'], nyear['SF-BMB'], nyear['SF-EE'] = 201

###### Defined Functions
def anu(var):
    # To calculate annual means and select years after 1900 
    return var.groupby('time.year').mean(dim='time')

def check_time(var):
    # the time coordinate of original file read is one month later than the actual time
    time = pd.date_range("1850-01-01", freq="M", periods=var.time.shape[0])
    return var.assign_coords(time=time)
  
def read_data(exp, identifier, variables = vars, fdir = fdir_atm):
    data = {}
    for v in variables:
        if (exp[:2]=='LE'):
          fname = v+'/b.e21.*.f09_g17.LE2-'+identifier+'.cam.h0.'+v+'.*.nc'
        else:
          fname = v+'/b.e21.*.f09_g17.CESM2-'+exp+'*'+identifier+'.cam.h0.'+v+'.*.nc'
        data[v] = (anu(check_time(xr.open_mfdataset(fdir+fname,combine='by_coords')[v]))).load()
    data['SST'] = data['SST'].where(data['SST']!=0)
    return data

def spna_avg(var, SPNA_domain_boundary = SPNA_domain_boundary):
    sel_var = var.where((var.lat-SPNA_domain_boundary[0])*(var.lat-SPNA_domain_boundary[1])<=0,drop=True).where((var.lon-SPNA_domain_boundary[2])*(var.lat-SPNA_domain_boundary[3])<=0,drop=True)
    return sel_var.mean(dim=('lat','lon'))

def glb_avg(var):
    # To calculate area-weighted global means
    weight = np.cos(np.deg2rad(var.lat))
    return var.weighted(weight).mean(('lat','lon'))

def cli_m(var, first_N_years = first_N_years):
    # To calculate climatologies
    return var[:first_N_years,:,:].mean('year')

def ano(var):
    # Return anomaly, defined as the deviation from climatology over 1850-1880
    return var-cli_m(var)

def Decompose_SSTA(data):
    # return local budget and global budget
    nyear = len(data['SST'].year)
    spna_ta = np.zeros((8,nyear))
    glm_ta = np.zeros((8,nyear))
    
    # total SSTA
    spna_ta[-1,:] = spna_avg(ano(data['SST']))
    glm_ta[-1,:] = glb_avg(ano(data['SST']))
    
    scaling_factor = 4*5.670373e-8*cli_m(data['SST'])**3 # coefficient 
    alb = (data['FSDS']-data['FSNS'])/data['FSDS'] # surface albedo 

    # SAF
    spna_ta[0,:] = spna_avg(-ano(alb)*(data['FSDS'])/scaling_factor)
    glm_ta[0,:] = glb_avg(-ano(alb)*(data['FSDS'])/scaling_factor)
    
    # LW CRF
    spna_ta[1,:] =  spna_avg(ano(data['FLDS']-data['FLDSC'])/scaling_factor)
    glm_ta[1,:] =  glb_avg(ano(data['FLDS']-data['FLDSC'])/scaling_factor)
 
    # SW CRF
    spna_ta[2,:] =  spna_avg((1-cli_m(alb))*ano(data['FSDS']-data['FSDSC'])/scaling_factor)
    glm_ta[2,:] =  glb_avg((1-cli_m(alb))*ano(data['FSDS']-data['FSDSC'])/scaling_factor)

    # clear-sky SW
    spna_ta[3,:] =  spna_avg((1-cli_m(alb))*ano(data['FSDSC'])/scaling_factor)
    glm_ta[3,:] =  glb_avg((1-cli_m(alb))*ano(data['FSDSC'])/scaling_factor)
  
    # clear-sky LW
    spna_ta[4,:] =  spna_avg(ano(data['FLDSC'])/scaling_factor)
    glm_ta[4,:] =  glb_avg(ano(data['FLDSC'])/scaling_factor)
     
    # Q (OHT div + dOHC/dt)
    spna_ta[5,:] =  spna_avg(-ano((data['FSNS']-data['FLNS']-data['SHFLX']-data['LHFLX']))/scaling_factor)
    glm_ta[5,:] =  glb_avg(-ano((data['FSNS']-data['FLNS']-data['SHFLX']-data['LHFLX']))/scaling_factor)
    
    # SH&LH
    spna_ta[6,:] =  spna_avg(-ano((data['SHFLX']+data['LHFLX']))/scaling_factor)
    glm_ta[6,:] =  glb_avg(-ano((data['SHFLX']+data['LHFLX']))/scaling_factor)
    
    return spna_ta,glm_ta


######### Decomposition 
spna_ssta = np.zeros((len(ens_idx[exp]), 8, nyear[exp])) # [irlzn,iterm,iyear]
glb_ssta = np.zeros((len(ens_idx[exp]), 8, nyear[exp]))
spna_ssta[:,:,:] = np.nan
glb_ssta[:,:,:] = np.nan

k=0
for idx in ens_idx[exp]:
    print(k)
    data = read_data(exp,idx)
    spna_ssta[k,:,:],glb_ssta[k,:,:] = decompose_ssta(data)
    k=k+1

######### Save data in a txt file 
np.savetxt(fdir_out+'CESM2-'+exp+'_decomposed_SPNASSTA_combined.txt', spna_ssta.flatten(), fmt='%7.5f')
np.savetxt(fdir_out+'CESM2-'+exp+'_decomposed_GMSSTA_combined.txt', glb_ssta.flatten(), fmt='%7.5f')
