import os
import pandas as pd
import numpy as np
from statistics import mean, stdev
from scipy.stats import norm
from datetime import datetime
import urllib3

# Désactiver les avertissements urllib3
urllib3.disable_warnings()

# Définir le chemin de travail
path = os.getcwd()

# Lire les fichiers CSV
inputs = pd.read_csv(os.path.join(path, 'inputs.csv'), encoding='utf-8-sig')
df_TradeLog = pd.read_csv(os.path.join(path, 'TradeLog.csv'))
df_PD = pd.read_csv(os.path.join(path, 'PD.csv'))

# Calculer les valeurs de base
Pclassique = inputs['Classique'].iloc[-1]
Pprecarite = inputs['Précarité'].iloc[-1]
#PriceSpot = mean([Pclassique, Pprecarite]) * 1000
PriceSpot = (Pclassique* 0.7 +  Pprecarite * 0.3) * 1000
Vclassique = stdev(inputs['R Classique'][1:]) * np.sqrt(52)
Vprécarité = stdev(inputs['R Précarité'][1:]) * np.sqrt(52)
VSpot = mean([Vclassique, Vprécarité])
r = 0.03
N = 1000000
today = pd.Timestamp(datetime.now().date())

# Filtrer les données de TradeLog
df_open = df_TradeLog[df_TradeLog['O/C'] == 'open'].copy()
df_Topen = df_open[df_open['Trade_Type'] == 'Trade'].copy()
df_Topen.rename(columns={'Expected_Delivery': 'Exp_Date','FINAL_BUYER': 'BUYER','BUY UNIT PRICE': 'B Price','SELL UNIT PRICE': 'S Price','Updated Expected Delivery': 'Up_Exp_Date'}, inplace=True)

df_Topen['Up_Exp_Date'] = pd.to_datetime(df_Topen['Up_Exp_Date'])
df_Topen['Maturity'] = (df_Topen['Up_Exp_Date'] - today).dt.days / 365.25

# Fusionner avec df_PD
df_PD_S = df_PD[['Counterparty','PD_Str']].rename(columns={'Counterparty': 'SELLER'})
df_PD_B = df_PD[['Counterparty','PD_Str']].rename(columns={'Counterparty': 'BUYER'})
df_Xopen = pd.merge(pd.merge(df_Topen, df_PD_S, how='left', on='SELLER'), df_PD_B, how='left', on='BUYER')
df_Xopen.rename(columns={'PD_Str_x': 'PD_S', 'PD_Str_y': 'PD_B'}, inplace=True)

# Calcul des VaR et ES pour chaque ligne de df_Topen
def calc_metrics(row):
    T = row['Maturity']
    Volume = row['QUANTITY']
    PriceB = row['B Price']
    PriceS = row['S Price']
    PD_B = row['PD_B']
    PD_S = row['PD_S']
    
    if Volume == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    rand_vals = norm.ppf(np.random.rand(N))
    exp_term = np.exp((r - 0.5 * Vclassique ** 2) * T + Vclassique * np.sqrt(T) * rand_vals)
    
    MtMB = Volume * (PriceSpot * exp_term - PriceB)
    MtMS = Volume * (PriceS - PriceSpot * exp_term)
    
    valeurs = np.concatenate((MtMB, MtMS))
    
    VaR = np.percentile(valeurs, 10)
    ES = valeurs[valeurs <= VaR].mean()
    
    MtMB_PD = MtMB * PD_B
    MtMS_PD = MtMS * PD_S
    valeurs_PD = np.concatenate((MtMB_PD, MtMS_PD))
    
    X_VaR = np.percentile(valeurs_PD, 10)
    X_ES = valeurs_PD[valeurs_PD <= X_VaR].mean()
    
    return [
        VaR, np.percentile(MtMB, 10), np.percentile(MtMS, 10),
        ES, MtMB[MtMB <= np.percentile(MtMB, 10)].mean(), MtMS[MtMS <= np.percentile(MtMS, 10)].mean(),
        X_ES, MtMB_PD[MtMB_PD <= np.percentile(MtMB_PD, 10)].mean(), MtMS_PD[MtMS_PD <= np.percentile(MtMS_PD, 10)].mean()
    ]

metrics = df_Xopen.apply(calc_metrics, axis=1, result_type='expand')
metrics.columns = ['VAR', 'VAR_Buyer', 'VAR_Seller', 'CVAR', 'CVAR_Buyer', 'CVAR_Seller', 'XVAR', 'XVAR_Buyer', 'XVAR_Seller']
df_Xopen = pd.concat([df_Xopen, metrics], axis=1)

Total_VAR = df_Xopen['VAR'].sum()
Total_ES = df_Xopen['CVAR'].sum()
Total_xES = df_Xopen['XVAR'].sum()

print(Total_VAR, Total_ES, Total_xES)

df_Xopen.to_csv('VAR_ES.csv', index=False)
