"""Trading signal generation from a spread series."""
import numpy as np
import pandas as pd

def zscore_signal(spread,window=30,entry_z=2.0,exit_z=0.5):
 mu=spread.rolling(window).mean(); sigma=spread.rolling(window).std().replace(0,np.nan); z=(spread-mu)/sigma
 pos=[]; cur=0
 for v in z:
  if pd.isna(v): pos.append(cur); continue
  if cur==0:
   if v>entry_z: cur=-1
   elif v<-entry_z: cur=1
  elif abs(v)<exit_z: cur=0
  pos.append(cur)
 return z,pd.Series(pos,index=spread.index)
