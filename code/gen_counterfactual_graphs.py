'''
Objective : This takes in as input data generated from counterfactual simulations and generates
            tables and graphs that go in the paper
Date      : January 13th, 2021

NOTE      : (1) DATA USED AS INPUT FOR THE CODE IS AVAILABLE UPON REQUEST
            (2) PATH NEEDS TO BE SET ON LINE 29
'''
#%%
import pandas as pd
import geopandas as gpd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from scipy import stats
import re
import glob
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
#%matplotlib qt

sns.set(color_codes=True)
sns.set_style("whitegrid",{'grid.linestyle': 'dotted'})

## SET PATH HERE
home = "input path here" 
os.chdir(home)

# Setting Folders
raw = home+'analysis/output/estimation/'
der = 'derived/output/'
outfol = home + 'analysis/output/model_analysis'

#%%
#------------------------------#
#  Functions
#------------------------------#
# Function: export text to file
def write_tex(filename,code):
    ''' Writes a Latex file given a filename and the underlying table code'''
    text_file = open(filename, "w")
    n = text_file.write(code)
    text_file.close()
    
# Set graph settings
sns.set(font='Palatino')
sns.set_style('white',{'font':'Palatino'})
def set_font(x):
    plt.rc('font',**{'family':'serif','serif':['Palatino']})
set_font(1)
snsblue = sns.color_palette()[0]
snsorg = sns.color_palette()[1]


#%%
# =============================================================================
#  Find Optimal Parameters
# =============================================================================

allshifts = list(range(1,7))
params = pd.read_csv(raw+'final_parameters.csv').replace(0,np.nan) # Import parameters
params.index = allshifts

# Import SE
se = pd.DataFrame({s:[np.sqrt(i) for i in np.diag(pd.read_csv(raw+'varcov_param_'+str(s)+'.csv'))] for s in allshifts[2:]},index=params.columns).T
    
# Get start and end hour for each shift
th = pd.read_excel(der+ 'estimation_data.xlsx', 'ids_time').groupby('shift')
th = th.head(1)[['shift','time']].merge(th.tail(1)[['shift','time']],on='shift',suffixes=['_start','_end']
                ).set_index('shift').applymap(lambda x: int(x[:2])).loc[allshifts,:]
th['time_end'] = th['time_end']+1 # last period is X:55 -> want X+1
th = th.applymap(lambda x: str(x)+' a.m.' if 0<x<12 else (str(x-12)+ ' p.m.' if x>12 else ('12 p.m.' if x==12 else '12 a.m.') ))
th['time'] = th['time_start']+" -- "+th['time_end']
th['name'] = th.index.map({3:"Afternoon",4:"Evening",5:"Night",6:"Late Night"}) # add shift nickname

# Export to tex table
paramtab = params.dropna().applymap(lambda x: format(x,'.2f')) # drop shifts not done yet
paramtab = th[['name','time']].merge(paramtab,left_index=True,right_index=True) # add shift name and time
paramtab = paramtab.append(se.applymap(lambda x: '('+format(x,'.3f')+')'),sort=False).sort_index().fillna('')
paramtab.columns = ['\textbf{Shift}','\textbf{Time}']+['$\\boldsymbol{'+c+'}$' for c in ['\sigma_\\varepsilon','\mu','\sigma_\eta']]
write_tex(outfol+'final_parameters.tex',paramtab.to_latex(index=False,escape=False,column_format='llrrr'))

#%%
# =============================================================================
#  Import Estimation Data
# =============================================================================
moments = pd.read_excel(der+ 'estimation_data.xlsx', 'moments').set_index('index')
regions = pd.read_excel('derived/output/estimation_data.xlsx', 'areas')
nreg = regions.region.max()
city = gpd.read_file('raw/GIS/neighborhoods.shp')
key_vals = pd.read_excel(der+ 'estimation_data.xlsx', 'key_vals')
time_ids = pd.read_excel(der+ 'estimation_data.xlsx', 'ids_time')
entryall = pd.read_excel(der+ 'estimation_data.xlsx', 'entry')
areas = pd.read_excel(der+ 'estimation_data.xlsx', 'areas')
eyeballs = pd.read_excel(der+ 'estimation_data.xlsx', 'eyeballs').drop_duplicates()

nontgt_data = pd.read_excel(der+ 'estimation_data.xlsx', 'nontarget')
nontgt_data.index = nontgt_data.index+1 # index represents the shift

# =============================================================================
#  Setup
# =============================================================================
eventsdfs = {}
run_shifts = range(3,7)

for shift in run_shifts:
    out =  outfol + 'shift' + str(shift) + '/'

    if shift<=4:
        sfx = '34'
    else:
        sfx = '56'
    dd_fe = pd.read_excel(der+ 'estimation_data.xlsx', 'demand_fe_shift'+sfx)
    dd_coeffs = pd.read_excel(der+ 'estimation_data.xlsx', 'demand_coeffs_shift'+sfx).values
    
    
    #%%
    # =============================================================================
    #  Importing estimation data
    # =============================================================================
    
    # Event DF Column Names
    colnames = ["evtype","driver","start_time","end_time","disp_time",
                          "start_hex","end_hex","disp_hex","fare","surge",
                          "orig_fare","ring","pull"]
    # Creating all Model names
    cfs_all = [str(s)+"_"+str(m)+"_"+ str(e) for s in ["SG","NS"] for m in ["MT","NM"] for e in ["EE","NE"]]
    title_map = {'SG_MT_EE':'Benchmark','SG_MT_NE':'Fixed RS',
              'SG_NM_EE':'No Match','SG_NM_NE':'Surge Taxi',
              'NS_MT_EE':'No Surge','NS_MT_NE':'Match Taxi',
              'NS_NM_EE':'Flexible Taxi','NS_NM_NE':'Taxi'}
    cfs_title_all = [title_map[c] for c in cfs_all]
    
    # Storing DFs for all cfs_all
    dat_all = {}
    vfs_all = {}
    unm_all = {}
    wd_all = {}
    mmts_all = {}
    
    for cf in cfs_all:
        ## EVENT DATA CODEBOOK
        # event_type: 0=entry, 1=trip, 2=search, 3=exit
        df = pd.read_csv(raw + cf + '_events_' + str(shift) + '.csv')
        df.columns = df.columns[:0].tolist() + colnames
        v = pd.read_csv(raw + cf + '_value_' + str(shift) + '.csv').values
        unmet = pd.read_csv(raw + cf + '_unmet_' + str(shift) + '.csv')
        waitdrop = pd.read_csv(raw + cf + '_waitdrop_' + str(shift) + '.csv')
        mmts = pd.read_csv(raw + cf + '_moments_' + str(shift) + '.csv')
        
        dat_all.update({cf:df})
        vfs_all.update({cf:v})
        unm_all.update({cf:unmet})
        wd_all.update({cf:waitdrop})
        mmts_all.update({cf:mmts})
    
    # Saving the events dataframe for the benchmark
    eventsdfs[shift] = dat_all['SG_MT_EE'].copy()
    
    # CFS to focus on for graphs
    cfs = ['SG_MT_EE','NS_MT_EE','SG_NM_EE','NS_NM_NE']
    cfs_title = [title_map[c] for c in cfs]
    
    # Creating list for relevant cfs
    dat = [dat_all[i] for i in cfs]
    vfs = [vfs_all[i] for i in cfs]
    unm = [unm_all[i] for i in cfs]
    wd = [wd_all[i] for i in cfs]
    mmts = [mmts_all[i] for i in cfs]
    
    # Get parameters for this shift
    sigma,mu,sigma_e = pd.read_csv(raw+'final_parameters.csv').iloc[shift-1,:].values
        
    #%%
    # =============================================================================
    #  Computing Non Spatial Moments
    # =============================================================================
    
    ## OUTPUTS
    tgt_mm = {'Pull':[],'SearchTime':[],'DrivCount':[]}
    tgt_mm.update({e+str(r):[] for e in ['Entry','Exit'] for r in range(1,nreg)})
        
    non_mm = {'Revenue':[],'FracSurge':[],'UnmetDd':[],'Exit':[],'SearchDist':[],'NumRides':[]}
    non_mm.update({e+str(r):[] for e in ['Search'] for r in range(1,nreg+1)})
    
    ## Computing Moments in the Data
    for i in range(len(dat)):
        df = dat[i]
        entry = df.loc[df.evtype==0].copy()
        rides = df.loc[df.evtype==1].copy()  
        search = df.loc[df.evtype==2].copy()
        exits = df.loc[df.evtype==3].copy()
        
        #------------------------------#
        #   Targeted Moments 
        #------------------------------#
        mm = mmts[i] # moments already computed in Julia
        
        # Average Pull
        tgt_mm['Pull'].append(mm.iloc[0,0])
        
        # Vacant Searching Times
        tgt_mm['SearchTime'].append(mm.iloc[1,0])
        # Manually calculate: - will use search later
        search['new_spell'] = search['start_time'] != search.groupby('driver')['end_time'].shift(1)
        search['spell'] = search.groupby('driver')['new_spell'].cumsum() # spell of searches
        vactime = search.groupby(['driver','spell']).agg({'evtype':'count','start_time':'min','end_time':'max'}
                                                         ).rename({'evtype':'count'},axis=1).reset_index()
        first_t = rides.groupby('driver')['disp_time'].min().rename('start_time').reset_index() # first time obs in data
        last_t = rides.groupby('driver')['end_time'].max().reset_index() # last time obs in data
        vactime = vactime.merge(first_t,on='driver',validate='m:1',suffixes=('','_first')
                        ).merge(last_t,on='driver',validate='m:1',suffixes=('','_last'))
        vactime['unobs'] = (vactime['end_time']<=vactime['start_time_first'])|(vactime['start_time']>=vactime['end_time_last']) # unobserved in data
        avgvac = vactime.loc[~vactime['unobs'],'count'].mean()  # drop first and last search to match the data 
        #tgt_mm['SearchTime'].append(avgvac)
    
        # Drivers 
        tgt_mm['DrivCount'].append(mm.iloc[2,0])
        
        # Entry/Exits by Region
        for r in range(1,nreg):
            tgt_mm['Entry'+str(r)].append(mm.iloc[2+r,0]) # 3-4
            tgt_mm['Exit'+str(r)].append(mm.iloc[4+r,0]) # 5-6
    
        
        #------------------------------#
        #   Non-Targeted Moments 
        #------------------------------#
    
        # Driver Revenues
        fares = rides.groupby('driver')['fare'].sum().reset_index()
        fares = fares['fare'].mean() # Output
        non_mm['Revenue'].append(fares)
        
        # Fraction Surged Rides
        non_mm['FracSurge'].append(100*(rides['surge']>1).mean())
        
        # Unmet Demand 
        unmet = pd.read_csv(raw + cfs[i] + '_unmet_'+ str(shift) + '.csv').values.sum()
        unmet = unmet/(unmet+rides.shape[0]) # convert to fraction of all requests
        non_mm['UnmetDd'].append(100*unmet)   
        
        # Last Rides
        last_ride = rides.groupby('driver').tail(1).groupby('end_hex')['evtype'].count().rename('last').reset_index()
        tot_end = rides.groupby('end_hex')['evtype'].count().rename('tot_end').reset_index()    
        if 'NE' in cfs[i]:
            non_mm['Exit'].append(0.00)
        else:        
            non_mm['Exit'].append(round(100*(last_ride['last'].sum()/tot_end['tot_end'].sum()),2))
        
        # Search Distance 
        non_mm['SearchDist'].append(search.groupby(['driver','spell'])['ring'].sum().mean())
            
        # Number of Rides
        non_mm['NumRides'].append(rides.shape[0])
        
        # Search Prob
        regsrch = search.groupby('end_hex').evtype.count().rename('search').reset_index(
                 ).merge(regions,left_on='end_hex',right_on='hex',how='left'
                 ).groupby('region')['search'].sum()
        regsrch = regsrch/regsrch.sum()
    
        for r in range(1,nreg+1):
            if r in regsrch.index:
                srchval = regsrch[regsrch.index==r].iloc[0]
            else:
                srchval = np.nan
            non_mm['Search'+str(r)].append(srchval)
                
       
    #%%
    # =============================================================================
    #  Moment Tables
    # =============================================================================        
    
    # Targeted Moments
        # rows order as in tgt_mm.keys()
    tgt_tit = dict(zip(tgt_mm.keys(),
                  ['Average Pull Distance','Vacant Searching Times','Drivers With Trips']+
                  ['\textit{P}('+e+' in Region '+str(r)+')' for e in ['Entry','Exit']for r in range(1,nreg)]))
    tgt_data = dict(zip(tgt_mm.keys(),moments[shift]))
    
    tgt_tab = []
    cc = ['Moment','Data']+cfs_title
    mm = list(tgt_mm.keys())
    for i in range(len(mm)):
        mr = mm[i]
        tgt_tab.append([tgt_tit[mr],tgt_data[mr]]+tgt_mm[mr])
    tgt_tab = pd.DataFrame(tgt_tab,columns=cc).set_index('Moment').applymap(lambda x: '{:.2f}'.format(x))
    for c in ['DrivCount']: # fix integer rows
        tgt_tab.loc[tgt_tit[c],:] = tgt_tab.loc[tgt_tit[c],:].apply(lambda x: '{:,}'.format(int(float(x))))
    
    
    # Non-Targeted Moments
        # rows order as in non_mm.keys()
    non_tit = dict(zip(non_mm.keys(),
                  ['Average Earnings','Surged Rides (\%)','Unmet Demand (\%)','Last Rides (\%)',
                   'Average Search Distance','Number of Trips']+
                  ['\textit{P}(Search in Region '+str(r)+')' for r in range(1,nreg+1)]))
    nontgt_dataS = nontgt_data.rename({'earnings':'Revenue','exits':'Exit','numrides':'NumRides',
                                       'fracsurge':'FracSurge'},axis=1).loc[shift]
    non_data = {} # add in the non-targeted moments from the data (sometimes missing)
    for k in non_mm.keys():
        if k in nontgt_dataS:
            non_data.update({k:nontgt_dataS[k]})
        else:
            non_data.update({k:'--'})
    
    non_tab = []
    cc = ['Moment','Data']+cfs_title
    mm = list(non_mm.keys())
    for i in range(len(mm)):
        mr = mm[i]
        non_tab.append([non_tit[mr],non_data[mr]]+non_mm[mr])
    non_tab = pd.DataFrame(non_tab,columns=cc).set_index('Moment').applymap(lambda x: '{:.2f}'.format(x) if type(x)==float else x).astype(str).applymap(lambda x: '--' if x=='nan' else x)
    for c in ['NumRides']: # fix integer rows
        non_tab.loc[non_tit[c],:] = non_tab.loc[non_tit[c],:].apply(lambda x: '{:,}'.format(int(float(x))))
    
    # Export 
    tabfin = {0:[tgt_tab,'targeted_moments'],1:[non_tab,'nontarget_moments']}
    for i in range(len(tabfin)):
        tab = tabfin[i][0]
        tex = tab.to_latex(index=True,escape=False)
        tex = tex[:tex.find('Moment')]+tex[tex.find('\\midrule'):]
        for c in cc[1:]:
            tex=tex.replace(c,'\\textbf{'+c+'}')
        write_tex(out+tabfin[i][1]+'.tex',tex)
     
    #%%
    # =============================================================================
    #  Counterfactual Bar Graph
    # =============================================================================
    
    # ==============================#
    # Creating DataFrame for Plotting
    # ============================= #
    
    # Prepare data to plot             
    plotval = [non_mm['Revenue'],tgt_mm['SearchTime'],
              non_mm['Search1'],tgt_mm['DrivCount']]
    plttit = ['Earnings','Search Time','Central Search','Num. of Drivers']
    
    plotdf = []
    rr = ['Benchmark','No Surge','No Matching']
    for i in range(len(plotval)):
        for j in range(len(rr)):
            val = 100*(plotval[i][j]/plotval[i][-1])
            plotdf.append([plttit[i],rr[j],val])
    plotdf = pd.DataFrame(plotdf,columns=['mmt','cf','val'])
    palette = ['indianred','cornflowerblue','darkgrey']
    
    # Plot bar graph
    with sns.axes_style("whitegrid"):
        set_font(1)
        fig, ax = plt.subplots()
        sns.barplot(x="val",y="mmt",hue="cf",data=plotdf,ax=ax,palette = palette)
        _=ax.set(xlabel='Percent Relative to Taxi Equilibrium',ylabel="")
        _=plt.legend(title='', loc='center',bbox_to_anchor=(0.5, 1.08),ncol=len(rr))
        _=plt.axvline(100,ls='--',lw=2,color='black')
        fig.savefig(out+'cf_comparison.png', format='png', dpi=600, bbox_inches = "tight")
    
    #%%
    # =============================================================================
    #  Spatial Plots
    # =============================================================================
    
    # Import shapefiles
    city = gpd.read_file(der+'hexagons/final_areas.shp')
    rephex = gpd.read_file(der+'hexagons/rep_hexes.shp')
    hexsrg = regions.set_index('hex')[['areaID']].to_dict()['areaID'] # hex to surge area dict
    mapx = city[city['central']==1].dissolve(by='central').centroid.iloc[0].x - 0.02 # get centre of area for title
    mapy = 30.49
    
    # Prepare map of side-by-side cases
    xoffset = 0.33
    yoffset = -0.44
    offset = [(i*xoffset,j*yoffset) for j in range(2) for i in range(2)]
    mapgrid = []
    maptits = []
    for i in range(len(offset)):
        gdf = city.copy()
        gdf.geometry = gdf.geometry.translate(xoff=offset[i][0],yoff=offset[i][1])
        gdf['model'] = cfs[i]
        mapgrid.append(gdf)
        
        ttx = mapx+offset[i][0]
        tty = mapy+offset[i][1]
        maptits.append([cfs_title[i],ttx,tty])
      
    cityhex = pd.concat(mapgrid)
    
        
    #%%
    # Prepare Data for Spatial Plotting (hex-level)
        # Surge
        # Unmet demand
        # Entry/Exit (first ride/last ride)
    
    ## Computing Moments in the Data
    spL = []
    for i in range(len(dat)):
        df = dat[i]
        
        rides = df.loc[df.evtype==1].copy()    
        tot_rides = rides.groupby('start_hex')['evtype'].count().rename('total').reset_index()
        tot_rides['start_frac'] = tot_rides['total']/tot_rides['total'].sum()
        tot_rides['lg_start'] = np.log(tot_rides['total'])
        
        # Surge    
        srg = rides.groupby('start_hex')['surge'].mean().reset_index()
            
        # Unmet
        unmet = pd.read_csv(raw + cfs[i] + '_unmet_'+ str(shift) + '.csv').sum(axis=1).reset_index(
                ).rename({'index':'start_hex',0:'unmet'},axis=1)
        unmet['start_hex'] = unmet['start_hex']+1
        unmet = unmet.merge(tot_rides,on='start_hex',how='outer').fillna(0)
        unmet['unmet'] = (1+unmet['unmet'])/(1+unmet['total']+unmet['unmet'])
        
        # Entry/Exit
        start_hex = 'disp_hex' # could be disp_hex instead
        first_ride = rides.groupby('driver').head(1).groupby(start_hex)['evtype'].count().rename('first').reset_index() # counts of where first rides start
        last_ride = rides.groupby('driver').tail(1).groupby('end_hex')['evtype'].count().rename('last').reset_index()
        tot_start = rides.groupby(start_hex)['evtype'].count().rename('tot_start').reset_index() # counts of where all rides start
        tot_end = rides.groupby('end_hex')['evtype'].count().rename('tot_end').reset_index()
        
        hexcnt = first_ride.merge(last_ride,left_on=start_hex,right_on='end_hex',how='outer'
                            ).merge(tot_start,on=start_hex,how='outer'
                            ).merge(tot_end,on='end_hex',how='outer') # put all counts together
        hexcnt['hex'] = hexcnt[start_hex].fillna( hexcnt['end_hex']).astype(int)
    
        hexcnt['entry'] = hexcnt['first']/hexcnt['tot_start'] # entry rate
        hexcnt['exit'] = hexcnt['last']/hexcnt['tot_end'] # exit  
        hexcnt = hexcnt[['hex','entry','exit']]
        
        sp = srg.merge(unmet,on='start_hex',how='outer'
                ).merge(tot_rides[['start_hex','start_frac']],how='outer'
                ).rename({'start_hex':'hex'},axis=1
                ).merge(hexcnt,on='hex',how='outer')
        sp['model'] = cfs[i]
        spL.append(sp)
        
    spdf = pd.concat(spL,axis=0)    
    mapdf = cityhex.merge(spdf,on=['hex','model'],how='left')
    
    #%%
    def plot4city(df,col,outname='cityplot',save=False,titles=maptits):
        # gives df and col to plot, prints out the 2x2 map and saves it in outname if save==True
        fig, ax = plt.subplots()
        misar = df[col].isna() # areas that are missing
        if misar.sum() > 0:
            df[misar].plot(ax=ax,color='gray',alpha=0.1,edgecolor='gray',linewidth=0.2)
        df[~misar].plot(ax=ax,column=col,cmap="YlOrRd",legend=True,legend_kwds={'shrink': 0.8},edgecolor='gray',linewidth=0.2)
        _= ax.axis('off')
        for i in range(4):
            _= plt.text(x=maptits[i][1],y=maptits[i][2],s=maptits[i][0],
                        ha='center',weight='demibold',fontsize=8)   
        plt.close()
        if save:
            fig.savefig(out+outname+'.png', format='png', dpi=600, bbox_inches = "tight")
        return fig
    
    for c in [c for c in spdf.columns.tolist() if c not in ['hex','model','total']]:
        plot4city(mapdf,c,'cityplot_'+c,save=True)
        
    #%%
    # =============================================================================
    #  Computing Welfare
    # =============================================================================
    
    #----------------------------------------------------#
    # Function to get Hour and Minute from Dispatch Time
    #----------------------------------------------------#
    
    # Start Hour is the hour in which the shift begins: for e.g. for shift 6 it is 0
    # time_ids = pd.read_excel(der+ 'estimation_data.xlsx', 'ids_time')
    tdf = time_ids[time_ids['shift']==shift].copy().reset_index(drop=True).reset_index() # time ids for this shift
    tdf['hour'] = tdf['time'].str[:2].apply(int)
    tdf['minute'] = tdf['time'].str[3:5].apply(int)
    tdf['t'] = tdf['index']+1
    get_time = tdf.set_index('t')[['hour','minute']].to_dict()
        # dict: get_time['hour'/'min'][time_id]
    
    #------------------------------#
    #           Setup 
    #------------------------------#
    T = tdf['t'].max()
    
    # Convert ring distance to minutes
    alpha = key_vals.loc[key_vals['param']=='alpha_time','value'].iloc[0]
    beta = key_vals.loc[key_vals['param']=='beta_time','value'].iloc[0]
    
    # Parameters for consumer welfare
    pmax = 7    # Upper bound for price
    wmax = 30   # Upper bound for waiting time
    pe = dd_coeffs[0,1] # Price Elasticity
    we = dd_coeffs[1,1] # Waiting Time Elasticity
    eyee = dd_coeffs[2,1] # Eyeballs elasticity
    fuel_cost = key_vals.loc[key_vals['param']=='fuel cost','value'].iloc[0]
    
    # Extracting Demand and Supply for shift
    time_rng = tdf['time_idx'].values # range of times in shift
    entry = entryall.loc[entryall['time_idx'].isin(time_rng),['demand','supply']].values
    
    #-------------------------------#
    # Function to Compute Welfare
    #-------------------------------#
    
    ## Inputs :  (1) List of Events Dataframes (2) List of Value Fn Matrices 
    #            (3) List of Unmet Demand DataFrames (4) List of CF labels  
    #            (5) CF to choose : 'benchmark','surge' or 'matching'
    
    ## Outputs : Consumer Surplus, Driver Revenue, Platform Revenue, Driver Revenues
    #                and collapsed Events data
    
    def Welfare(dat,vfs,unm,model):
        df = dat[model]
        V = vfs[model]
        u = unm[model]
        
        # ======================
        #     Driver Welfare
        # ======================
        # Never entered
        entered = df.loc[df.evtype==0].groupby('start_time').driver.count().values
        never_entered = (entry[0:len(entered),1] - entered).sum()
        ne_welfare = never_entered*T*mu
    
        # Pre Entry Welfare for those Entered at t
        pre_welfare = sum([entered[t]*mu*t for t in range(len(entered))])
    
        # Post Entry Welfare
        post = df.loc[df.evtype==0].groupby(['start_time','start_hex']).driver.count().reset_index()
        post['welfare'] = 0.0
        for index, row in post.iterrows():   
            t = int(row['start_time'].item())
            l = int(row['start_hex'].item())
            post.at[index,'welfare'] = V[l-1,t-1]*row['driver']
    
        post_welfare = post.welfare.sum()
        
        # Revenue
        fares = df.loc[df.evtype==1].fare.sum() # fares earned
        fuel_trip = (df.loc[df.evtype==1,['ring','pull']].sum(axis=1)*fuel_cost).sum() # fuel cost for trips
        fuel_search = (df.loc[df.evtype==2].ring*fuel_cost).sum() # fuel cost for driving
    
        # Total Driver Welfare and Revenue
        ds = ne_welfare + pre_welfare + post_welfare
        rev = fares-fuel_trip-fuel_search
        avgrev = rev/df['driver'].max() # divide by total drivers who enter
        
        # ======================
        #     Consumer Welfare
        # ======================
           
        # Merging with areas DF to be able to collapse
        rides = df.loc[df.evtype==1].merge(areas, how ='left', left_on ='start_hex',
                      right_on ='hex')
        
        rides['wait'] = alpha + beta*rides['pull'] # converting pull to minutes
        
        # Extracting 5-Minute Interval and Hour 
        rides['hour'] = rides['disp_time'].map(get_time['hour'])
        rides['minute'] =  rides['disp_time'].map(get_time['minute'])
        
        # Merging Unmet Demand with Events Data
        u = u.unstack().reset_index()
        u.columns = ['disp_time','hex','ud']
        u['hex'] +=1
        u['disp_time'] = u['disp_time'].apply(lambda x : int(x[1:]))
        u = u.merge(areas, how ='left', on ='hex').groupby(['areaID',
                                           'disp_time']).agg({'ud':'sum'}).reset_index() # Merging with Surge Areas
        u['hour'] = u['disp_time'].map(get_time['hour'])
        u['minute'] =  u['disp_time'].map(get_time['minute'])
            
        # Merging with Eyeballs data and FE data and Unmet Demand
        rides = rides.groupby(['areaID','disp_time']).agg({'surge':'mean','wait':'mean',
                                                             'fare':'mean','hour' : 'first',
                                                             'minute' : 'first','evtype':'count'}).reset_index(
                                                        ).merge(eyeballs,how = 'left',
                                                        on = ['areaID','hour','minute']).merge(dd_fe,
                                                        how = 'left', left_on = ['areaID','hour',
                                                        'minute'], right_on = ['areaID','hour','mt']).merge(
                                                        u,how = 'left',on = ['areaID','hour','minute'])
        
        # ================
        # 	Rider Welfare
        # ================   
        rides['surplus'] = np.exp(rides['hdfe'])*((pmax**(pe+1) - np.power(rides['surge'].values,pe+1))/(pe+1)
                                        )*((wmax**(we+1) - np.power(rides['wait'].values,we+1))/(we+1)
                                        )*np.power(rides['eyeballs'].values,eyee)*((rides['evtype'])/(rides['ud']+rides['evtype']))
        cs = rides.surplus.sum()
        
        # ==================
        # 	Platform Profit
        # ==================
        # This is just equal to the number of rides
        profit = len(df.loc[df.evtype==1])
            
        return cs,rev,avgrev,profit,ds,rides
    
    #%%
    # ============================================================================
    # 	Counterfactual Welfare for Benchmark, Surge, No Matching Relative to Taxi
    # ============================================================================
    
    # Welfare Calculations for All Models
    wf_vals = ['Consumer Surplus','Driver Revenue','Avg. Driver Revenue',
          'Platform Revenue','Producer Surplus']
    wf = {} # repeat dict for each model
    rides_data_all = {}
    
    for model in cfs_all:
        wel_output = Welfare(dat_all,vfs_all,unm_all,model)
        wf.update({model:dict(zip(wf_vals,wel_output[:5]))})
        rides_data_all.update({model:wel_output[5]})
        
    
    # Creating Dataframe to Plot
    base_model = 'Taxi'
    plot_dat = pd.DataFrame([wf[c] for c in cfs],index=cfs_title) # restrict to cfs
    plot_dat = plot_dat.apply(lambda x: 100*x/x.loc[base_model]).unstack().reset_index() # make relative to baseline
    plot_dat.columns = ['mm','cf','val']
    plot_dat = plot_dat.loc[(plot_dat['cf']!="Taxi") &
                            (plot_dat['mm']!="Producer Surplus")] # restrict which ones to plot
    
    
    ## Welfare Comparison Plot
    with sns.axes_style("whitegrid"):
        set_font(1)
        fig, ax = plt.subplots()
        sns.barplot(x="val",y="mm",hue="cf",data=plot_dat,ax=ax,palette = palette)
        _=ax.set(xlabel='Percent Relative to Taxi Equilibrium',ylabel="")
        _=plt.legend(title='', loc='center',bbox_to_anchor=(0.5, 1.08),ncol=3)
        _=plt.axvline(100,ls='--',lw=2,color='black')
        fig.savefig(out+'welfare_comparison.png', format='png', dpi=600, bbox_inches = "tight")
    
    #%%
    # ============================================================================
    # 	Counterfactual Welfare for All Policies Relative to Taxi
    # ============================================================================
    
    # Creating Dataframe to Plot
    base_model = 'Taxi' 
    cfs_order = [0,4,2,1,3,5,6,7]
    cfs_plot = [cfs_all[i] for i in cfs_order]
    cfs_title_plot = [cfs_title_all[i] for i in cfs_order]
    
    plot_dat = pd.DataFrame([wf[c] for c in cfs_plot],index=cfs_title_plot) # restrict to cfs
    plot_dat = plot_dat.apply(lambda x: 100*x/x.loc[base_model]).unstack().reset_index() # make relative to baseline
    plot_dat.columns = ['mm','cf','val']
    plot_dat = plot_dat.loc[(plot_dat['cf']!="Taxi") &
                            (~plot_dat['mm'].isin(["Producer Surplus","Platform Revenue"]))] # restrict which ones to plot
    
    
    # Data to Plot
    d1 = plot_dat.loc[plot_dat.mm=="Consumer Surplus"]
    d2 = plot_dat.loc[plot_dat.mm=="Driver Revenue"]
    d3 = plot_dat.loc[plot_dat.mm=="Avg. Driver Revenue"]
    
    ## Welfare Comparison Plot
    with sns.axes_style("whitegrid"):
        set_font(1)
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex = False,figsize=(10,15),
                                              gridspec_kw={'height_ratios':[1,1,1,0.1]})
        sns.barplot(y="val",x="cf",data=d1,ax=ax1,
                    palette = ['darkgrey' if (x < 100) else 'indianred' for x in d1.val.values])
        sns.barplot(y="val",x="cf",data=d2,ax=ax2,
                    palette = ['darkgrey' if (x < 100) else 'indianred' for x in d2.val.values])
        sns.barplot(y="val",x="cf",data=d3,ax=ax3,
                    palette = ['darkgrey' if (x < 100) else 'indianred' for x in d3.val.values])
        _=ax1.set(ylabel='Percent Relative to Taxi Equilibrium',xlabel="")
        _=ax2.set(ylabel='Percent Relative to Taxi Equilibrium',xlabel="")
        _=ax3.set(ylabel='Percent Relative to Taxi Equilibrium',xlabel="")
        _=ax1.set_title('Consumer Surplus',fontsize=18)
        _=ax2.set_title('Driver Revenue',fontsize=18)
        _=ax3.set_title('Avg. Driver Revenue',fontsize=18)
        _=ax1.axhline(100,ls='--',lw=2,color='black')
        _=ax2.axhline(100,ls='--',lw=2,color='black')
        _=ax3.axhline(100,ls='--',lw=2,color='black')
        
        # Table below graph
        rows = ["$\\bf{"+r+"}$" for r in ['Surge','Matching','Entry/Exit']]
        cell_text = []
        for i in range(len(rows)):
            row = []    
            for col in cfs_plot[:-1]:
                row.append(('N' not in col.split('_')[i]))
            cell_text.append(row)
        
        ax4.axis('off')
        the_table = ax4.table(cellText=cell_text,
                          rowLabels=rows,cellLoc='center',
                          loc='center',edges='horizontal')
        plt.subplots_adjust(left=0.2, bottom=-0.3)
    
        fig.savefig(out+'welfare_comparison_all.png', format='png', dpi=600, bbox_inches = "tight")
    
    #%%
    ## Complementarity Table
    base_model = 'Flexible Taxi'
    rs_cols = ["No Surge","No Match","Benchmark"]
    plot_dat = pd.DataFrame([wf[c] for c in cfs_plot],index=cfs_title_plot) # restrict to cfs
    plot_dat = plot_dat.apply(lambda x: 100*x/x.loc[base_model] - 100).unstack().reset_index() # make relative to baseline
    plot_dat.columns = ['mm','cf','val']
    plot_dat = plot_dat.loc[(plot_dat['cf'].isin(rs_cols)) &
                            (~plot_dat['mm'].isin(["Producer Surplus","Avg. Driver Revenue"]))] # restrict which ones to plot
    plot_dat = plot_dat.pivot(index='mm',columns='cf',values='val')[rs_cols]
    plot_dat['Complementarity'] = plot_dat.iloc[:,2] - (plot_dat.iloc[:,0]+plot_dat.iloc[:,1])
    plot_dat = plot_dat.applymap(lambda x: round(x,2)).reset_index().rename({'mm':'Welfare'},axis=1)
    plot_dat['Welfare'] = plot_dat['Welfare'].apply(lambda x: '\textbf{'+x+'}')
    plot_dat.columns = ['\textbf{'+x+'}' for x in plot_dat.columns] 
    
    write_tex(out+'complement_table.tex',plot_dat.to_latex(index=False,escape=False))
    
    
    base_model = 'Taxi'
    rs_cols = ["Match Taxi","Surge Taxi","Fixed RS"]
    plot_dat = pd.DataFrame([wf[c] for c in cfs_plot],index=cfs_title_plot) # restrict to cfs
    plot_dat = plot_dat.apply(lambda x: 100*x/x.loc[base_model] - 100).unstack().reset_index() # make relative to baseline
    plot_dat.columns = ['mm','cf','val']
    plot_dat = plot_dat.loc[(plot_dat['cf'].isin(rs_cols)) &
                            (~plot_dat['mm'].isin(["Producer Surplus","Avg. Driver Revenue"]))] # restrict which ones to plot
    plot_dat = plot_dat.pivot(index='mm',columns='cf',values='val')[rs_cols]
    plot_dat['Complementarity'] = plot_dat.iloc[:,2] - (plot_dat.iloc[:,0]+plot_dat.iloc[:,1])
    plot_dat = plot_dat.applymap(lambda x: round(x,2)).reset_index().rename({'mm':'Welfare'},axis=1)
    plot_dat['Welfare'] = plot_dat['Welfare'].apply(lambda x: '\textbf{'+x+'}')
    plot_dat.columns = ['\textbf{'+x+'}' for x in plot_dat.columns] 
    
    write_tex(out+'complement_table_fixed.tex',plot_dat.to_latex(index=False,escape=False))
    
    #%%
    # ===========================
    #  Spatial Consumer Surplus
    # ===========================
    
    cityareas = cityhex.dissolve(by = ['areaID','model']).reset_index()
    dfL = []
    for c in cfs:
        df = rides_data_all[c].groupby('areaID').surplus.sum().reset_index()
        df['surplus'] = np.log(1+df['surplus'])
        df['model'] = c
        dfL.append(df)
    mapareadf = cityareas.merge(pd.concat(dfL,axis=0),on=['areaID','model'],how='left')
    
    plot4city(mapareadf,'surplus','cityplot_surplus',save=True)
    
    
    #%%
    # ===========================
    #  Value Function Plot
    # ===========================
    
    # Time periods -> x axis
    TS = np.linspace(1,T,T)
    
    # Benchmark Value Functions
    v = vfs[cfs.index('SG_MT_EE')]
    # Smoothening the Value Function Matrix
    plotvecs = [np.polyval(np.polyfit(TS,v[i,0:len(TS)],2),TS) for i in range(len(v)-1)]
    
    ## Making Plot
    n = len(plotvecs)
    fig,ax = plt.subplots() #create figure and axes
    for i in range(n):
        ax.plot(TS,plotvecs[i],linestyle = 'dashed',lw = 1.0)
    _=plt.xlabel('Time Periods (5 minute intervals)')
    _=plt.ylabel('Value Functions')    
    fig.savefig(out+'value_func_benchmark.png', format='png', dpi=600, bbox_inches = "tight")

# ===========================
#  Surge Distribution Plot
# ===========================
# Distribution from Data
sdata3 = pd.DataFrame(data = {'value':['1.0','1.25','1.50','1.75','2.0',
                                      '2.25 +'],'freq':[0.81,0.084,
                                    0.072,0.018,0.01,0.006]})
sdata3['type'] = 'Data'            
sdata6 = pd.DataFrame(data = {'value':['1.0','1.25','1.50','1.75','2.0',
                                      '2.25 +'],'freq':[0.53,0.12,
                                    0.11,0.06,0.05,0.13]})
sdata6['type'] = 'Data'            
sdata = {3:sdata3,6:sdata6}

# Distribution from Model
smodel = {}
for v in [3,6]:
    dfs = eventsdfs[v]
    dfs = dfs.loc[dfs.surge>=1].surge.value_counts(normalize = True).rename_axis('value').reset_index(name='freq')
    dfs.loc[dfs.value>=2.25,'value'] = 2.25
    dfs = dfs.groupby('value').sum().reset_index()
    dfs['value'] = dfs.value.map({1:'1.0',1.25:'1.25',1.50:'1.50',1.75:'1.75',2:'2.0',2.25:'2.25 +'})
    dfs['type'] = 'Model'
    smodel[v] =  dfs

# Plotting Distribution
palette1 = ['indianred','cornflowerblue']
for s in [3,6]:
    plotdist = sdata[s].append(smodel[s])
    with sns.axes_style("whitegrid"):
        set_font(1)
        fig,ax = plt.subplots() #create figure and axes
        sns.barplot(x='value',y='freq',data=plotdist,hue='type',palette = palette1)
        sns.despine()
        _= plt.xlabel('Surge Factor',fontsize='small')
        _= plt.ylabel('Proportion',fontsize='small')
        _=plt.xticks(fontsize = 8)
        _=plt.yticks(fontsize = 8)
        _=plt.legend(fontsize = 9)
        legend = ax.legend(loc = 'upper right')
        fig = ax.get_figure()
        fig.savefig(outfol+'welfare/surgedist_'+str(s)+'.png', format='png', dpi=600, bbox_inches = "tight")

#%%    
# =============================================================================
#  Surge-Match ICs
# =============================================================================

out = home + 'analysis/output/model_analysis_' + date + '/welfare/'
HM = {}
CFstats = {}
dfs={}
for sfx in ['','_flexi','_comp']:
    # Import each shifts files
    allcombos = []
    for shf in range(3,7):
        combo = pd.read_csv(raw+'combo/combo_cfs_'+str(shf)+sfx+'.csv')
        combo['platform_profit'] = combo['platform_profit'] + combo['tot_rides']
        combo['driver_profit'] = combo['driver_profit'] - combo['tot_rides']
        combo['total_welfare'] = combo['consumer_surplus']+combo['driver_profit']+combo['platform_profit']
        
        wv = ['frac_surge','avg_surge','vacant','avg_pull',
                  'entrypr1','entrypr2','exitpr1','exitpr2']
        for v in wv: # varaibles that are weighted by rides
            combo[v] = combo[v]*combo['tot_rides']
        allcombos.append(combo)
 
    # Aggregate for the day
    combo = pd.concat(allcombos).groupby(['max_pull','surge_amp']).sum().reset_index()
    for v in wv: # re-weight by total rides
        combo[v] = combo[v]/combo['tot_rides']
    combo = combo.rename({'tot_rides':'total_rides',
                          'frac_surge':'fraction_surge',
                          'avg_surge':'average_surge',
                          'vacant':'vacant_time',
                          'actdriv':'driver_count',
                          'avg_pull':'average_pull'},axis=1)
    dfs[sfx] = combo.copy()

    # Determine variables to export
    outvars = ['consumer_surplus','driver_profit','platform_profit','total_welfare',
               'total_rides','driver_count',
               'average_surge','fraction_surge',
               'vacant_time','average_pull']
    
    benchmark = (combo['max_pull']==10)&(combo['surge_amp']==1.0)
        
    if sfx=='': # Get baseline value of variables
        sfx='_base' # for consistent file name
        baseval = {v:combo.loc[benchmark,v].iloc[0] for v in outvars}
        
    CFstats[sfx[1:]] = combo.loc[benchmark,outvars].assign(cf=sfx[1:])
    
    combo['CSimprove'] = (combo['consumer_surplus']>baseval['consumer_surplus'])
    combo['DPimprove'] = (combo['driver_profit']>baseval['driver_profit'])*1
    combo['TWimprove'] = (combo['total_welfare']>baseval['total_welfare'])*1
    combo['PPimprove'] = (combo['platform_profit']>baseval['platform_profit'])*1    
    
    bestCell = [combo.loc[combo[c].idxmax(),['max_pull','surge_amp']].to_list() 
                for c in ['consumer_surplus','driver_profit','total_welfare','platform_profit']]
    
    normalize=True # normalize the welfare values or not
    if normalize:
        for v in outvars:
            if v in ['average_surge','fraction_surge','average_pull']:
                continue # don't normalize these variables
            combo[v] = combo[v]/baseval[v]
    
    HM[sfx[1:]] = {}
    for outcome in outvars:
        hmdf = combo.pivot(index='surge_amp',columns='max_pull',values=outcome)
        if outcome in ['consumer_surplus','driver_profit','total_welfare']:
            HM[sfx[1:]][outcome] = hmdf
        if outcome in ['average_pull','average_surge','fraction_surge']:
            cc = baseval[outcome]
        else:
            cc = 1
        
        fig, ax = plt.subplots()
        sns.heatmap(hmdf,center=cc,cmap="seismic_r")
        _= plt.title(outcome.replace('_',' ').title())
        _= plt.ylabel('Surge Amplification')
        _= plt.xlabel('Max Pull Ring')
        _=plt.yticks(rotation=0, ha='right')
        _=ax.tick_params(axis='both', which='major', pad=-2)
        _=ax.invert_yaxis()
        _=ax.add_patch(Rectangle((9, 2), 1, 1, fill=False, edgecolor='gold', lw=3))
        fig = ax.get_figure()
        fig.savefig(out+'combo'+sfx+'_'+outcome+'.png', format='png', dpi=600, bbox_inches = "tight")
                
    # Changing one instrument
    with sns.axes_style("white"):
        set_font(1)
        fig, ax = plt.subplots()
        sns.despine()
        sns.lineplot(x='max_pull',y='consumer_surplus',data=combo[combo['surge_amp']==1],label='Consumer Surplus')
        sns.lineplot(x='max_pull',y='driver_profit',data=combo[combo['surge_amp']==1],label='Driver Profit')
        sns.lineplot(x='max_pull',y='total_welfare',data=combo[combo['surge_amp']==1],label='Total Welfare')
        sns.lineplot(x='max_pull',y='total_rides',data=combo[combo['surge_amp']==1],label='Total Rides')    
        _= plt.xlabel('Max Pull Ring')
        _= plt.ylabel('Relative to Baseline')
        _=plt.axvline(10,ls='--',lw=1,color='gray')
        _=plt.axhline(1.0,ls='--',lw=1,color='gray')
        fig = ax.get_figure()
        fig.savefig(out+'combo'+sfx+'_vary_pull.png', format='png', dpi=600, bbox_inches = "tight")
    
    with sns.axes_style("white"):
        set_font(1)
        fig, ax = plt.subplots()
        sns.despine()
        sns.lineplot(x='surge_amp',y='consumer_surplus',data=combo[combo['max_pull']==10],label='Consumer Surplus')
        sns.lineplot(x='surge_amp',y='driver_profit',data=combo[combo['max_pull']==10],label='Driver Profit')
        sns.lineplot(x='surge_amp',y='total_welfare',data=combo[combo['max_pull']==10],label='Total Welfare')
        sns.lineplot(x='surge_amp',y='total_rides',data=combo[combo['max_pull']==10],label='Total Rides')    
        _= plt.xlabel('Surge Amplification')
        _= plt.ylabel('Relative to Baseline')
        _=plt.axvline(1.0,ls='--',lw=1,color='gray')
        _=plt.axhline(1.0,ls='--',lw=1,color='gray')    
        fig = ax.get_figure()
        fig.savefig(out+'combo'+sfx+'_vary_amp.png', format='png', dpi=600, bbox_inches = "tight")
   
    gg = {0:{'x':'max_pull','c':'surge_amp','bx':10,'bc':1,'xlab':'Max Pull Ring'},
          1:{'x':'surge_amp','c':'max_pull','bx':1,'bc':10,'xlab':'Surge Amplification'}}
    with sns.axes_style("white"):
        set_font(1)
        fig, ax = plt.subplots(1,2,figsize=(10,4))
        sns.despine()  
        for i in [0,1]:
            sns.lineplot(x=gg[i]['x'],y='consumer_surplus',data=combo[combo[gg[i]['c']]==gg[i]['bc']],label='Consumer Surplus',ax=ax[i])
            sns.lineplot(x=gg[i]['x'],y='driver_profit',data=combo[combo[gg[i]['c']]==gg[i]['bc']],label='Driver Profit',ax=ax[i])
            sns.lineplot(x=gg[i]['x'],y='total_welfare',data=combo[combo[gg[i]['c']]==gg[i]['bc']],label='Total Welfare',ax=ax[i])
            sns.lineplot(x=gg[i]['x'],y='total_rides',data=combo[combo[gg[i]['c']]==gg[i]['bc']],label='Total Rides',ax=ax[i])    
            _= ax[i].set_xlabel(gg[i]['xlab'])
            _= ax[i].set_ylabel('Relative to Baseline')
            _= ax[i].axvline(gg[i]['bx'],ls='--',lw=1,color='gray')
            _= ax[i].axhline(1.0,ls='--',lw=1,color='gray')    
            #_= ax[i].set_ylim(0.5,2)
            _= ax[i].get_legend().remove()
    fig.legend(ax[i].get_legend_handles_labels()[0],ax[i].get_legend_handles_labels()[1],
               loc='center', bbox_to_anchor=(0.25, -0.35, 0.5, 0.5),ncol=2)
    fig.savefig(out+'combo'+sfx+'_vary.png', format='png', dpi=600, bbox_inches = "tight")
        
    with sns.axes_style("white"):
        set_font(1)
        fig, ax = plt.subplots(1,2,figsize=(10,4))
        sns.despine()  
        for i in [0,1]:
            ax1 = ax[i].twinx()
            sns.lineplot(x=gg[i]['x'],y='average_surge',data=combo[combo[gg[i]['c']]==gg[i]['bc']],label='Average Surge',ax=ax[i],color=snsblue)
            sns.lineplot(x=gg[i]['x'],y='average_pull',data=combo[combo[gg[i]['c']]==gg[i]['bc']],label='Average Pull',ax=ax1,color=snsorg)
            _= ax[i].set_xlabel(gg[i]['xlab'])
            _= ax[i].set_ylabel('Surge Factor')
            _= ax1.set_ylabel('Pull Ring')            
            _= ax[i].axvline(gg[i]['bx'],ls='--',lw=1,color='gray')
            if i==0:
                _= ax[i].set_ylim(1,1.4)
            else:
                _= ax1.set_ylim(1.8,2.3)
            _= ax[i].get_legend().remove()
            _= ax1.get_legend().remove()
    fig.legend(ax[i].get_legend_handles_labels()[0]+ax1.get_legend_handles_labels()[0],
               ax[i].get_legend_handles_labels()[1]+ax1.get_legend_handles_labels()[1],
               loc='center', bbox_to_anchor=(0.25, -0.25, 0.5, 0.5),ncol=2)
    fig.tight_layout(pad=2.2)
    fig.savefig(out+'combo'+sfx+'_vary_pp.png', format='png', dpi=600, bbox_inches = "tight")
    
    impgrid = {}
    for v in ['DP','TW','PP']:
        hmdf = combo.pivot(index='surge_amp',columns='max_pull',values=v+'improve')
        zm = np.ma.masked_less(hmdf.values, 0.5)
        x= np.arange(len(hmdf.columns)+1)
        y= np.arange(len(hmdf.index)+1)
        impgrid[v] = {'x':x,'y':y,'zm':zm}
    hmdf = combo.pivot(index='surge_amp',columns='max_pull',values='CSimprove')
    
    bestC = [(hmdf.columns.to_list().index(bestCell[i][0]),hmdf.index.to_list().index(bestCell[i][1])) for i in range(len(bestCell))]
    bestcolor = ['mediumblue','darkorange','mediumpurple','darkgreen']
    addbest = False
    
    # Pareto Improving plot
    cs_col = 'goldenrod'
    dp_col = 'whitesmoke'
    fig, ax = plt.subplots()
    sns.heatmap(hmdf,cmap=['white',cs_col], linewidths=.05, linecolor='lightgray',cbar=False)
    _= plt.title('Welfare Comparisons')
    _= plt.ylabel('Surge Amplification')
    _= plt.xlabel('Max Pull Ring')
    _=plt.yticks(rotation=0, ha='right')
    _=ax.tick_params(axis='both', which='major', pad=-2)
    _=ax.invert_yaxis()
    _=ax.add_patch(Rectangle((9, 2), 1, 1, fill=False, edgecolor='red', lw=3))
    if addbest == True:
        _=ax.add_patch(Rectangle(bestC[0], 1, 1, fill=False, edgecolor=bestcolor[0], lw=2))    
        _=ax.add_patch(Rectangle(bestC[1], 1, 1, fill=False, edgecolor=bestcolor[1], lw=2, ls='--'))    
        _=ax.add_patch(Rectangle(bestC[2], 1, 1, fill=False, edgecolor=bestcolor[2], lw=2, ls=':'))        
        _=ax.add_patch(Rectangle(bestC[3], 1, 1, fill=False, edgecolor=bestcolor[3], lw=2, ls='-.'))         
        bs_patch0 = mpatches.Patch(fill=False, edgecolor=bestcolor[0], lw=2, label='Max CS')
        bs_patch1 = mpatches.Patch(fill=False, edgecolor=bestcolor[1], lw=2, ls='--', label='Max DP')
        bs_patch2 = mpatches.Patch(fill=False, edgecolor=bestcolor[2], lw=2, ls=':', label='Max TW')
        bs_patch3 = mpatches.Patch(fill=False, edgecolor=bestcolor[3], lw=2, ls='-.', label='Max Profit')        
    tw=plt.pcolor(impgrid['TW']['x'], impgrid['TW']['y'], impgrid['TW']['zm'],
                  facecolor='blue', hatch='..', alpha=0.05)
    dp=plt.pcolor(impgrid['DP']['x'], impgrid['DP']['y'], impgrid['DP']['zm'],
                  facecolor=dp_col, hatch='//', alpha=0.05)
    pp=plt.pcolor(impgrid['PP']['x'], impgrid['PP']['y'], impgrid['PP']['zm'],
                  facecolor=dp_col, hatch='\\\\', alpha=0.05)    
    bs_patch = mpatches.Patch(fill=False, edgecolor='red', lw=3, label='Benchmark')
    cs_patch = mpatches.Patch(color=cs_col, label='Cons. Surplus $\\uparrow$')
    dp_patch = mpatches.Patch(facecolor=mc.to_rgba(dp_col), edgecolor=(0,0,0,1), lw=0, hatch='///', label='Driver Profit $\\uparrow$')    
    pp_patch = mpatches.Patch(facecolor=mc.to_rgba(dp_col), edgecolor=(0,0,0,1), lw=0, hatch='\\\\', label='Platform Profit $\\uparrow$')          
    tw_patch = mpatches.Patch(fill=False, edgecolor=(0,0,0,1), lw=0.5, hatch='...', label='Total Welfare $\\uparrow$')
    if addbest == True:    
        plt.legend(handles=[bs_patch,cs_patch,dp_patch,pp_patch,tw_patch,
                            bs_patch0,bs_patch1,bs_patch2,bs_patch3], bbox_to_anchor=(1.01, 1), loc='upper left')    
    else:
        plt.legend(handles=[bs_patch,cs_patch,dp_patch,pp_patch,tw_patch], bbox_to_anchor=(1.01, 1), loc='upper left')    
    fig = ax.get_figure()
    fig.savefig(out+'combo'+sfx+'_pareto.png', format='png', dpi=600, bbox_inches = "tight")

 
# Combined heatmap
for v in HM['base'].keys():
    dfL = [HM['base'][v],HM['comp'][v],HM['flexi'][v]]
    lims = (pd.concat(dfL).min().min(),pd.concat(dfL).max().max())    
    comboCFs = ['Benchmark','Platform Commission','Flexible Surge']
    fig, axn = plt.subplots(1, 3, sharey=False,sharex=True,figsize=(12,12))
    im = []
    for i, ax in enumerate(axn.flat):
        g = sns.heatmap(dfL[i], ax=ax,center=1,cmap="seismic_r",
                cbar = False, vmin=lims[0], vmax=lims[1])
        im.append(g)
        _= ax.set_title(comboCFs[i])
        _= ax.set(xlabel='Max Pull Ring')
        if i==0:
            _= ax.set(ylabel='Surge Amplification')
            _=ax.add_patch(Rectangle((9, 2), 1, 1, fill=False, edgecolor='gold', lw=3))
        else:
            _= ax.set(ylabel='')
        _= ax.set_yticklabels(ax.get_yticklabels(),rotation=0, ha='right')
        _= ax.tick_params(axis='both', which='major', pad=-2)
        _= ax.invert_yaxis()
        _= ax.set_aspect(1.0)
        
    mappable = im[0].get_children()[0]
    plt.colorbar(mappable, ax = axn,orientation = 'horizontal',fraction=0.025, pad=0.07,
                 label='Relative to Benchmark')
    fig = ax.get_figure()
    fig.savefig(out+'allcombo_'+v+'.png', format='png', dpi=600, bbox_inches = "tight")

# Table
dfst = pd.concat([CFstats['base'],CFstats['comp'],CFstats['flexi']]).set_index('cf').T
dfst = dfst.applymap(lambda x: '{:,.2f}'.format(x))
for c in ['total_rides','driver_count']:
    dfst.loc[c,:] = dfst.loc[c,:].apply(lambda x: x[:-3])
dfst.columns = ['\textbf{'+c+'}' for c in comboCFs]
dfst.index = [i.replace('_',' ').title() for i in dfst.index]
write_tex(out+'welfare_flexicomp.tex',dfst.to_latex(escape=False,column_format='lrrr'))


#---------------------------------
# Plot for Solving Inefficiencies
#---------------------------------
## Unmet Demand

# Getting the Match Taxi Unmet Demand
ud = 0
tr = 0
# Event DF Column Names
colnames = ["evtype","driver","start_time","end_time","disp_time",
            "start_hex","end_hex","disp_hex","fare","surge",
            "orig_fare","ring","pull"]
for s in range(3,7):
    ud += pd.read_csv(raw + 'NS_MT_NE' + '_unmet_'+ str(s) + '.csv').values.sum()
    trdf = pd.read_csv(raw + 'NS_MT_NE' + '_events_' + str(s) + '.csv')
    trdf.columns = trdf.columns[:0].tolist() + colnames
    tr += trdf[trdf.evtype==1].shape[0]

# Creating Unmet Demand DataFrame
unmetdd = {v:dfs[''].loc[(dfs['']['max_pull']==v) & (dfs['']['surge_amp']==1.0) ,'unmet_dem'].iloc[0] for v in [10,1]}
unmetdd['ns'] = dfs[''].loc[(dfs['']['max_pull']==10) & (dfs['']['surge_amp']==0.0) ,'unmet_dem'].iloc[0]
unmetdd['mt'] = ud
totdd = {v:dfs[''].loc[(dfs['']['max_pull']==v) & (dfs['']['surge_amp']==1.0) ,'total_rides'].iloc[0] for v in [10,1]}
totdd['ns'] = dfs[''].loc[(dfs['']['max_pull']==10) & (dfs['']['surge_amp']==0.0) ,'total_rides'].iloc[0]
totdd['mt'] = tr
unmetdd = {v : unmetdd[v]/(unmetdd[v]+totdd[v]) for v in unmetdd}
unmetdd['Benchmark'] = unmetdd.pop(10)
unmetdd['No Matching'] = unmetdd.pop(1)
unmetdd['No Surge'] = unmetdd.pop('ns')
unmetdd['No Surge (Fixed SS)'] = unmetdd.pop('mt')
unmetdd = pd.DataFrame(list(unmetdd.items()),columns = ['Model','Unmet Demand'])

# Vacant Search Time
vactime = {v:dfs[v].loc[(dfs[v]['max_pull']==10) & (dfs[v]['surge_amp']==1.0) ,'vacant_time'].iloc[0] for v in ['','_flexi']}
vactime['nomatch'] = dfs[''].loc[(dfs['']['max_pull']==1) & (dfs['']['surge_amp']==1.0) ,'vacant_time'].iloc[0]
vactime['nomatchflexi'] = dfs['_flexi'].loc[(dfs['_flexi']['max_pull']==1) & (dfs['_flexi']['surge_amp']==1.0) ,'vacant_time'].iloc[0]
vactime['Benchmark'] = vactime.pop('')
vactime['No Match (BM)'] = vactime.pop('nomatch')
vactime['Flexi Surge'] = vactime.pop('_flexi')
vactime['No Match (Flexi Surge)'] = vactime.pop('nomatchflexi')
vactime = pd.DataFrame(list(vactime.items()),columns = ['Model','Vacant Search Time'])

# Plotting
palette = ['indianred','cornflowerblue','darkgrey','olivedrab']
with sns.axes_style("white"):
    set_font(1)
    fig, ax = plt.subplots()        
    sns.despine()  
    plt.bar(list(range(0,4)),unmetdd['Unmet Demand'],width= 0.65,color = palette)
    plt.xticks(list(range(0,4)), ['Benchmark','No Match','No Surge','No Surge (Fixed SS)'])
    _=plt.ylabel('Proportion of Unmet Demand',fontsize = 'small')
    _=plt.xlabel('')
    _=plt.title('Static Inefficiency')    
    axes = plt.gca()
    axes.yaxis.grid()
    fig.savefig(outfol+'welfare/static_ineff.png', format='png', dpi=600, bbox_inches = "tight")

with sns.axes_style("white"):
    set_font(1)
    fig, ax = plt.subplots()        
    sns.despine()  
    plt.bar(list(range(0,4)),vactime['Vacant Search Time'],width= 0.65,color = palette)
    plt.xticks(list(range(0,4)), ['Benchmark','No Match','Flexi Surge','No Match (Flexi)'])
    _=plt.ylabel('Vacant Search Time (5 min intervals)',fontsize = 'small')
    _=plt.xlabel('')    
    _=plt.title('Dynamic Inefficiency')        
    axes = plt.gca()
    axes.yaxis.grid()
    fig.savefig(outfol+'welfare/dynamic_ineff.png', format='png', dpi=600, bbox_inches = "tight")
    
#%%
# =============================================================================
#  Pricing CFs
# =============================================================================

allshifts = range(3,7)

# Import pricing search files
allvals = []
for s in allshifts:
    gridfiles = glob.glob(raw+'pricing/pricing_cfs_'+str(s)+'_*')
    if len(gridfiles)>0:
        vals=[]
        for f in gridfiles:
            vals.append(pd.read_csv(f))
        vals = pd.concat(vals)
        vals = vals[vals['base_rate']>0]# drop if not done
        vals['shift'] = s
        allvals.append(vals)
        
pricing = pd.concat(allvals).reset_index(drop=True)  

# Removing rides from driver profit
pricing.loc[pricing.comp_rate==1.0,'driver_profit'] = pricing.loc[pricing.comp_rate==1.0,
                                                   'driver_profit'] -pricing.loc[pricing.comp_rate==1.0,'tot_rides']
pricing.to_csv(raw+'pricing.csv',index=False)

# Set baseline
baseopts = {'base_rate':1.5,'dist_rate':1.0,'surge_amp':1.0,'flexi_surge':False,'comp_rate':1.0}
params = list(baseopts.keys())
welvars = ['consumer_surplus','driver_profit','platform_profit','tot_rides']

# Aggregate shifts to entire day
pdf = pricing.groupby(params)[welvars].sum().reset_index().sort_values('platform_profit',
                                                         ascending=False).reset_index(False)
pdf['total_welfare'] = pdf[welvars[:3]].sum(1)
pdf['c'] = 'comp_'+(pdf['comp_rate']).astype(str)

# Calculating Benchmark Values
basevars=['consumer_surplus','driver_profit','tot_rides']
cond = (pdf['base_rate']==1.5)&(pdf['dist_rate']==1.0)&(pdf['surge_amp']==1.0)&(
        pdf['flexi_surge']==False)&(pdf['comp_rate']==1.0)
bmarkvals = {v:pdf.loc[cond,v].iloc[0] for v in basevars}
bmarkvals['platform_profit']=bmarkvals.pop('tot_rides')

# ---------------------------------------------------------
# Creating Table with Optimal parameters for CS, DP and PP
# ---------------------------------------------------------
tabvars=['consumer_surplus','driver_profit','platform_profit']
optim = {}
for v in tabvars:
    optim[v]=list(pdf.loc[pdf[v].idxmax(),[v]+params])
    optim[v][0] = round((optim[v][0]/bmarkvals[v] - 1)*100,2)  

optim = pd.DataFrame(optim)
optim.iloc[4] = ['Yes','Yes','Yes']
optim[''] = ['Increase from Benchmark (\%)','Base Rate (\$1.5)','Distance Rate (\$1.00)','Surge Amplification (1)','Flexible Surge (No)','Driver Compensation (1)']
optim = optim.rename(columns={'consumer_surplus':'\textbf{Consumer Surplus}',
                                 'driver_profit':'\textbf{Driver Profit}',
                                 'platform_profit':'\textbf{Platform Profit}'})
optim = optim[['','\textbf{Consumer Surplus}','\textbf{Driver Profit}','\textbf{Platform Profit}']]
write_tex(outfol+'optim_pricing.tex',optim.to_latex(index=False,escape=False,column_format='lccc'))

# ----------------------------------------------
# Plotting Wedge Induced by Profit Max Platform
# ----------------------------------------------

basevars=['consumer_surplus','driver_profit']
labels = ['Consumer Optimal','Driver Optimal']

# Computing Distortion under Benchmark Prices
cond = (pdf['base_rate']==1.5)&(pdf['dist_rate']==1.0)&(pdf['surge_amp']==1.0)&(
        pdf['flexi_surge']==False)&(pdf['comp_rate']==0.75)
distort = {v:(bmarkvals[v]/pdf.loc[cond,v].iloc[0])*100 for v in basevars}
distort = pd.DataFrame(distort.items(),columns=['var','percent'])
distort['Prices'] = 'Benchmark'

# Computing Distortion under Consumer Optimal Prices
for var in basevars:
    optimvals =  {v:pdf.loc[pdf[var].idxmax(),v] for v in basevars}
    dat = pdf.loc[pdf[var].idxmax(),:]
    profitvals = {v:(optimvals[v]/pdf.loc[(pdf.base_rate==dat[1])&
                         (pdf.dist_rate==dat[2])&
                         (pdf.surge_amp==dat[3])&
                         (pdf.flexi_surge==dat[4])&
                         (pdf.comp_rate==0.75),v].iloc[0])*100 for v in basevars}
    optim_distort = pd.DataFrame(profitvals.items(),columns = ['var','percent'])
    optim_distort['Prices'] = labels[basevars.index(var)]
    distort = distort.append(optim_distort)

# Appending Dataset
distort['var'] = distort['var'].map({'consumer_surplus':'Consumer Surplus',
                                      'driver_profit':'Driver Profit'})

# Making Plot
palette = ['indianred','cornflowerblue','darkgrey']
plt.rcParams['legend.title_fontsize'] = 'small'
with sns.axes_style("whitegrid"):
    set_font(1)
    fig, ax = plt.subplots()        
    sns.despine()  
    sns.barplot(y="percent",x="var",hue="Prices",data=distort,ax=ax,palette = palette)
    _=plt.ylabel('Social Planner relative to Profit Maximizing Platform',fontsize = 'small')
    _=plt.xlabel('')    
    _=plt.legend(title='Prices', loc='upper right',bbox_to_anchor=(1.5, 1))
    _=plt.axhline(100,ls='--',lw=2,color='black')
    fig.savefig(outfol+'welfare/distortion.png', format='png', dpi=600, bbox_inches = "tight")

    
#%%

    
    
