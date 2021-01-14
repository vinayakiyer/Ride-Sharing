#----------------------------------------------------------------------------#
"""
Project : Ride Austin  
Date Modified : January 13th, 2021

 1. This script constructs dataset for implementing a border discontinuity to estimate PED.
     The running variable here is the distance to the nearest border of a surge area
 
"""
#----------------------------------------------------------------------------#
#%%
# import packages 

import os 
import geopandas as gpd 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from datetime import datetime
import numpy as np
from geopy.distance import geodesic,lonlat,distance
import shapely
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.linestring import LineString
from shapely.geometry.multilinestring import MultiLineString
import multiprocessing
from joblib import Parallel, delayed

#%matplotlib qt5 

# Seaborn options
sns.set(color_codes=True)
sns.set_style("whitegrid",{'grid.linestyle': 'dotted'})

#%%
# =============================================================================
#  Setup
# =============================================================================

# set working directory 
home = ''  
der='derived/output/'
os.chdir(home)

#%%
# =============================================================================
#  Importing Data
# =============================================================================

#----------------#
# Shapefiles     #
#----------------#
shp = gpd.read_file(home + 'raw/GIS/surge_areas/surge_areas.shp')

# Adding Neighboring Surge Areas column
shp["nbrs"] = None 
for index, row in shp.iterrows():   
    # get nieghbors
    neighbors = shp[shp.geometry.touches(row['geometry'])].areaID.tolist()
    shp.at[index, "nbrs"] = neighbors

#----------------#
# Rides File     #
#----------------#
rides = pd.read_pickle(der+'rides_final.pickle')

## Merging Shapes files to Ride File
rides = pd.merge(rides,shp,how = 'left', left_on = 'areaID_start', 
                 right_on = 'areaID')

#%%
# =============================================================================
#  Preparing Data
# =============================================================================

# Subsetting data to where areaID geometry is present and is in-sample
rides = rides[rides['sample']].dropna(subset = ['geometry'])

# Creating Lat-Longs GeoDataFrame from Start Location
latlong = [Point(xy) for xy in zip(rides.start_long, rides.start_lat)]
crs = {'init' :'epsg:4326'}
cols_to_keep = ['rideID','nbrs','areaID_start']
latlong = gpd.GeoDataFrame(rides[cols_to_keep],crs=crs, geometry = latlong)
latlong.to_crs(epsg=3857,inplace=True) # Converting so that distance is in metres

# Converting the Shapefile to same CRS
shp.to_crs(epsg=3857,inplace=True) # Converting so that distance is in metres

#--------------------------------------------#
# Computing Distance to nearest neighbor     #
#--------------------------------------------#

# Find areas near surge area boundaries
bndry = shp.copy()
bndry['x'] = 1
bndry['geometry'] = bndry.geometry.boundary.buffer(2000)
bndry = bndry.dissolve(by='x')


## Calculating distances
def calcdist(i,shp,latlong):    
    row=latlong.loc[i]
    
    # Distances to all the neighbors
    d = shp[shp['areaID'].isin(row['nbrs'])].geometry.exterior.distance(row['geometry']).tolist()
    
    # Finding the minimum distance and the corresponding surge area
    pos = d.index(np.nanmin(d))
    mindist=np.nanmin(d)  # Minimum Distance
    nbrID =  row['nbrs'][pos] # Corresponding Surge Area
    
    return [i,mindist,nbrID]

inputs = latlong.index.tolist()
dists = Parallel(n_jobs=-1,verbose=10,prefer="threads")(delayed(calcdist)(i,shp,latlong) for i in inputs)

# Create distance dataframe
distsdf = pd.DataFrame(dists,columns=['index','mindist','nbrID']).sort_values('index').set_index('index')
latlong = pd.concat([latlong,distsdf],axis=1)

# Concatenating with Rides Dataset
rides = pd.concat([rides,latlong[['mindist','nbrID']]],axis = 1)

rides.to_pickle('temp.pickle')

#%%
# =============================================================================
#  Preparing Data for the RD
# =============================================================================

#------------------------------------------------#
# Preparing data to Assign Treatment and Control #
#------------------------------------------------#

## Computing FE for Surge Areas sharing a border

# Unique border pair ID
    # 6 digit string: [areaID1][areaID2], where areaID1<areaID2
rides['areaID'] = rides['areaID'].astype(int)
reorder = rides['areaID']>rides['nbrID']
fe1 = (rides['areaID']*1000 + rides['nbrID']).apply(lambda x: str(x).zfill(6))
fe2 = (rides['nbrID']*1000 + rides['areaID']).apply(lambda x: str(x).zfill(6))
rides['borderFE'] = fe1
rides.loc[reorder,'borderFE'] = fe2[reorder]


## Calculating Grouped Time for Ride Requests
def grouptime(df,var,interval):
    return df[var].apply(lambda dt: dt if pd.isnull(dt) else datetime(dt.year,
             dt.month, dt.day, dt.hour,interval*(dt.minute //interval)))

# Calculating intervals for Trip Request Time
for q in range(5,25,5):
    rides['trip_request_'+str(q)] = grouptime(rides,'rider_request',q).dt.time

## Creating Bins of Distance
bins = np.linspace(0,10000,51)
rides['bin'] = pd.cut(rides['mindist'],bins)
rides = rides.sort_values(by=['mindist'])
rides['distbin'] = rides.groupby('bin').ngroup() + 1

## For each cell computing: average surge, total demand, average wait time
groupcols = ['areaID','borderFE','distbin','trip_date_norm','trip_request_15']
ridedf = rides.groupby(groupcols).agg({'surge':'mean','count':'sum','rider_wait':'mean'}
                                      ).reset_index().rename({'count':'dd'},axis=1)

#-------------------------------------------------------------------#
# Collapsing Data to Border x SurgeArea x Distance x 15min interval #
#-------------------------------------------------------------------#

# Create a balanced panel
bordercols = ['borderFE','trip_date_norm','trip_request_15']
bt = ridedf[bordercols].drop_duplicates() # border-time obs that have dd>0 on at least one side 
bt['areaID1'] = bt['borderFE'].str[0:3].astype(int)
bt['areaID2'] = bt['borderFE'].str[3:].astype(int)
bt = pd.melt(bt,id_vars=bordercols,value_name='areaID').drop('variable',axis=1)

# Get surge history info
surge_history = pd.read_pickle(der+'surge_history.pickle')
surge_history['trip_request_15'] = grouptime(surge_history,'time',15).dt.time
histcols = ['areaID','trip_date_norm','trip_request_15']
surge_history = surge_history.groupby(histcols)['surge_rec'].mean().reset_index()

# Add surge info to balanced panel
surge_data = ridedf.groupby(bordercols+['areaID'])['surge'].mean().reset_index()
surgedf = bt.merge(surge_data,on=bordercols+['areaID'],how='left',validate='1:1'
            ).merge(surge_history,on=histcols,how='left',validate='m:1'
            ).rename({'surge':'surge_data','surge_rec':'surge_hist'},axis=1)
surgedf['surge'] = surgedf['surge_data'].fillna(surgedf['surge_hist'])

# Add distance bins to balanced panel
bb = pd.DataFrame([[b,1] for b in range(1,11)],columns=['distbin','x']) # bins 1-10
surgedf['x'] = 1
df = surgedf.merge(bb,on='x').drop('x',axis=1)

# Add demand
df = df.merge(ridedf[bordercols+['areaID','distbin','dd','rider_wait']],
                  on=bordercols+['areaID','distbin'],how='left',validate='1:1')
df['dd']= df['dd'].fillna(0).astype(int)
                    
## Calculating Treatment variable
df['near_dist'] = df['distbin']<=5
price = 'surge' # choose which surge variable to use [surge,surge_data,surge_hist]
assign = df[df['near_dist']].groupby(bordercols+['areaID'])[price,'dd'].agg({price:'mean','dd':'sum'}
             ).reset_index().rename({price:'surge'},axis=1)
assign['side'] = 1+1*(assign['borderFE'].str[0:3].astype(int)!=assign['areaID'])

assignS = pd.pivot_table(assign,index=bordercols,columns='side',values='surge').reset_index().rename({1:'surge1',2:'surge2'},axis=1)
assignD = pd.pivot_table(assign,index=bordercols,columns='side',values='dd').reset_index().rename({1:'dd1',2:'dd2'},axis=1)
assign = assignS.merge(assignD,on=bordercols,validate='1:1')

assign['side'] = 1+1*(assign['surge1']<assign['surge2']) # 1 for side1, 2 for side2 being larger
assign = assign.loc[(assign['surge1']!=assign['surge2'])&(assign[['surge1','surge2']].notna().all(axis=1))] 
    # drop where sides have equal surge or if surge is missing
assign['treatID'] = assign.apply(lambda x: int(x['borderFE'][(x['side']-1)*3:(x['side']-1)*3+3]),axis=1)
    # extract areaID from the border fixed effect ID
for c in ['surge','dd']:
    assign[c+'T'] = assign[c+'1']*(assign['side']==1) + assign[c+'2']*(assign['side']==2)
    assign[c+'C'] = assign[c+'1']*(assign['side']!=1) + assign[c+'2']*(assign['side']!=2)
assign = assign.drop(['surge1','surge2','dd1','dd2','side'],axis=1).reset_index(drop=True)

## Finalize dataset
df = df.merge(assign,on=bordercols,how='inner',validate='m:1')
df['treatment'] = (df['areaID']==df['treatID'])*1
    
## Calculating the Running Variable
df['dist'] = df['distbin']*(df['treatment']==1) - df['distbin']*(df['treatment']==0)

df['dow'] = df['trip_date_norm'].dt.dayofweek
df['hour'] = df['trip_request_15'].astype(str).str[0:2].astype(int)
df['trip_request_15'] = df.groupby('trip_request_15').ngroup()

## Exporting to STATA
df.to_stata(der+'rd_data.dta',write_index=False)


