# -*- coding: utf-8 -*-
"""

Student name: Jay Bhavesh Doshi
Student matriculation number: 4963577
              

"""
# %%
# Aim: To find the optimized capacity of solar, on-shore and off-shore wind
# to achieve the desired renewable energy share. 
#%%
import numpy as np
import pandas as pd #import pandas to work with dataframes

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

import matplotlib 
import matplotlib.pyplot as plt
# %%
weatherData = pd.read_csv('input/weather-data_wind-pv_Freiburg.csv', index_col=0, parse_dates = True)

capacityFactors = pd.read_csv('input/renewable_cf2015.csv', index_col=0)

data = pd.read_csv('input/demand2015.csv', index_col=0) #read csv file as dataframe (read documentation for more details)
data = data.set_index(capacityFactors.index)

installedSolarCapacity = 53110 #MW
installedOnWindCapacity = 54490 #MW
installedOffWindCapacity = 7740 #MW

#%% Varying the percentage of different RE cost at a time with specific RE share and storage capacity, 4 different 
# scenarios can be obtained with extra RE installations and new investment cost and % of curtailed power

renewableShareTarget = 0.8 #0.6 for another case

solarCost = 398e-3 #mln EUR per MW installed capacity (varying 20% higher and lower)
windOnshoreCost = 1118e-3 #mln EUR per MW installed capacity (varying 20% higher and lower)
windOffshoreCost = 2128e-3 #mln EUR per MW installed capacity (varying 20% higher and lower)
storageCost = 232e-3 #mln EUR per MWh installed capacity

storageCapacity = 500e3 #in MWh (2000GWh for another cse)
storagePower = 10.8e3 #in MW (updating w.r.t change in capacity)
chargingEfficiency = 0.82
dischargingEfficiency = 0.92
initialSOC = 0.5 #initial State of Charge (ratio from capacity)

current_investment = (installedSolarCapacity*solarCost + installedOnWindCapacity*windOnshoreCost
 + installedOffWindCapacity*windOffshoreCost +  storageCost*storageCapacity)
# %%
def RenShareTargetOpt(data, capacityFactors):
    model = pyo.ConcreteModel()
    
    model.i = pyo.RangeSet(0, len(data)-1)
        
    model.solarCapacity = pyo.Var(domain=pyo.NonNegativeReals, bounds = (0.0, 300e3))
    model.windOnshoreCapacity = pyo.Var(domain=pyo.NonNegativeReals, bounds = (0.0, 300e3))
    model.windOffshoreCapacity = pyo.Var(domain=pyo.NonNegativeReals, bounds = (0.0, 300e3))
    
    model.renGen = pyo.Var(model.i, domain=pyo.NonNegativeReals)
    
    model.batteryCapacity = pyo.Var(domain=pyo.NonNegativeReals)
    model.SOC = pyo.Var(model.i, domain=pyo.NonNegativeReals)
    model.charge = pyo.Var(model.i, domain=pyo.NonNegativeReals, bounds = (0.0, storagePower))
    model.discharge = pyo.Var(model.i, domain=pyo.NonNegativeReals, bounds = (0.0, storagePower))
    
    model.conventionalGen = pyo.Var(model.i, domain=pyo.NonNegativeReals)
    model.renShare = pyo.Var(model.i, domain=pyo.NonNegativeReals)
    
    model.investmentCost = pyo.Var(domain=pyo.NonNegativeReals)
    model.curtailment = pyo.Var(model.i, domain=pyo.NonNegativeReals)

    def renGen_rule(model, i):
        return model.renGen[i] == (model.solarCapacity + installedSolarCapacity) * capacityFactors['solar'].iloc[i] \
            + (model.windOnshoreCapacity + installedOnWindCapacity) * capacityFactors['onshore'].iloc[i] \
            + (model.windOffshoreCapacity + installedOffWindCapacity) * capacityFactors['offshore'].iloc[i]
            
            
    def SOC_rule(model, i):
        if i == 0:
            return model.SOC[i] == initialSOC * (storageCapacity + model.batteryCapacity) + model.charge[i] * chargingEfficiency - model.discharge[i] / dischargingEfficiency
        else:
            return model.SOC[i] == model.SOC[i-1] + model.charge[i] * chargingEfficiency - model.discharge[i] / dischargingEfficiency
               
    
    def energyBalance_rule(model, i):
        return data['demand'].iloc[i] + model.curtailment[i] + model.charge[i] == model.conventionalGen[i] + model.renGen[i] + model.discharge[i]
    
    
    def renShare_rule(model, i):
        return model.renShare[i] == 1 - model.conventionalGen[i] / data['demand'].iloc[i]
    
    
    def investmentCost_rule(model):
        return model.investmentCost == solarCost*model.solarCapacity + windOnshoreCost*model.windOnshoreCapacity + windOffshoreCost*model.windOffshoreCapacity \
            + storageCost*model.batteryCapacity
    
    
    def renShareTarget_rule(model):
        return ((renewableShareTarget-0.001), pyo.summation(model.renShare)/len(data), (renewableShareTarget+0.001))
    
    
    def batteryCapacity_rule(model, i):
        return model.SOC[i] <= model.batteryCapacity + storageCapacity
        

    
    model.renGen_rule = pyo.Constraint(model.i, rule=renGen_rule)
    model.SOC_rule = pyo.Constraint(model.i, rule=SOC_rule)
    model.energyBalance_rule = pyo.Constraint(model.i, rule=energyBalance_rule)
    model.renShare_rule = pyo.Constraint(model.i, rule=renShare_rule)
    model.investmentCost_rule = pyo.Constraint(rule=investmentCost_rule)
    model.renShareTarget_rule = pyo.Constraint(rule=renShareTarget_rule)
    model.batteryCapacity_rule = pyo.Constraint(model.i, rule = batteryCapacity_rule)
    
    def ObjRule(model):
        return model.investmentCost
     
    model.obj = pyo.Objective(rule=ObjRule, sense=pyo.minimize)
    
    opt = SolverFactory("glpk")
    
    opt.solve(model)
    
    return model

def get_values(model):
    renShare = []
    convGen = []
    curtailed = []  
    renGen = []
    
    for i in range(len(data)):
        renShare.append(model.renShare[i].value)
        convGen.append(model.conventionalGen[i].value)
        curtailed.append(model.curtailment[i].value)
        renGen.append(model.renGen[i].value)

    return renShare, convGen, curtailed, renGen

# %%

model = RenShareTargetOpt(data, capacityFactors)

# %%
renShare, convGen, curtailed, renGen = get_values(model)

solarProduction = (installedSolarCapacity + model.solarCapacity.value) * capacityFactors['solar']
windOffshoreProduction = (installedOffWindCapacity + model.windOffshoreCapacity.value) * capacityFactors['offshore']
windOnshoreProduction = (installedOnWindCapacity + model.windOnshoreCapacity.value) * capacityFactors['onshore']

data["renewableGeneration"] = solarProduction +  windOnshoreProduction +  windOffshoreProduction#create a new column with renewable generation

curtailedPercentage = sum(curtailed) / sum(renGen) * 100

print('Renewable share', round(np.mean(renShare),2)*100, '%')
print('Extra Solar capacity:', round(model.solarCapacity.value/1000, 0), 'GW')
print('Extra Wind Onshore capacity:', round(model.windOnshoreCapacity.value/1000, 0), 'GW')
print('Extra Wind Offshore capacity:', round(model.windOffshoreCapacity.value/1000, 0), 'GW')
print('Extra Battery Storage capacity:', round(model.batteryCapacity.value/1000, 0), 'GWh')
print('Total investment:', round(model.investmentCost.value/1000,1), 'billion EUR')

print("Curtailed: ", round(curtailedPercentage, 2), '%')

# %% Plotting the results

fig, ax = plt.subplots()
labels = ['Solar \n capacity(MW)', 
          'Onshore Wind \n capacity(MW)', 
          'Offshore Wind \n capacity(MW)', 
          'Battery \n capacity(MWh)', 
          'Investment \n (million EUR)']
x = np.arange(len(labels))  # the label locations
width = 0.35 

old_share = [installedSolarCapacity, installedOnWindCapacity, 
             installedOffWindCapacity, storageCapacity, current_investment]
new_share = [round(model.solarCapacity.value, 0), 
             round(model.windOnshoreCapacity.value, 0), 
             round(model.windOffshoreCapacity.value, 0), 
             round(model.batteryCapacity.value, 0), 
             round(model.investmentCost.value,1) ]

rects1 = ax.bar(x - width/2, old_share, width, label='Before optimization')
rects2 = ax.bar(x + width/2, new_share, width, label='After optimization')

ax.set_ylabel('(MW, MWh, mil EUR)')
ax.set_title('Comparing current and optimized installation')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7)
ax.legend()


fig.tight_layout()

plt.show()