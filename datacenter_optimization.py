# -*- coding: utf-8 -*-
"""

Student name: Jay Bhavesh Doshi , Anna Lebowsky
Student matriculation number: 4963577 ,
              

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
# TODO Aggregate all wind turbine generation in one-column: wind_gen = wind_1r[kW] + wind_2r[kW] + wind_3r[kW] + wind_4r[kW]

capacityFactors = pd.read_csv('input/renewable_cf2015.csv', index_col=0, parse_dates= True) # cf from python-lecture_12, also 2019 available

data = pd.read_csv('input/demand2015.csv', index_col=0, parse_dates= True) # cf from python-lecture_12, also 2019 available
data = data.set_index(capacityFactors.index)

# installedSolarCapacity = 53110 #MW
# installedWindCapacity = 54490 #MW

renewableShareTarget = 1 #0.6 for another case

# solarCost = 398e-3 #mln EUR per MW installed capacity (varying 20% higher and lower)
# windOnshoreCost = 1118e-3 #mln EUR per MW installed capacity (varying 20% higher and lower)
# windOffshoreCost = 2128e-3 #mln EUR per MW installed capacity (varying 20% higher and lower)
# storageCost = 232e-3 #mln EUR per MWh installed capacity

# #Solar
# Area_PV = 0 #m^2 #TODO Values TO_BE_CHECKED
# eff_PV = 0.7 #TODO Values TO_BE_CHECKED
#
# #Wind
# ratedPower_Wind = 0 #MW #TODO Values TO_BE_CHECKED

#Battery
self_discharge_rate = 0 #TODO Values TO_BE_CHECKED
storageCapacity = 500e3 #in MWh (2000GWh for another cse)
storagePower = 10.8e3 #in MW (updating w.r.t change in capacity)
chargingEfficiency = 0.82
dischargingEfficiency = 0.92
initialSOC = 0.5 #initial State of Charge (ratio from capacity)
SOCmin = 0 #TODO Values TO_BE_CHECKED
SOCmax = 0 #TODO Values TO_BE_CHECKED

#H2_electrolyzer = Hydrogen storage
hydrogen_operating_Pmin = 0 #MW #TODO Values TO_BE_CHECKED
hydrogen_operating_Pmax = 0 #MW #TODO Values TO_BE_CHECKED
H2_electrolyzer_eff = 0.8 #TODO Values TO_BE_CHECKED
HHV_H2 = 0 #TODO Values TO_BE_CHECKED

#H2_fuel cell = Hydrogen generation
hydrogen_powerGen_max = 10 #MW #TODO Values TO_BE_CHECKED
H2_fuelcell_eff = 0.8 #TODO Values TO_BE_CHECKED
LHV_H2 = 0 #TODO Values TO_BE_CHECKED

#Hydrogen_tank
hydrogen_tank_capacity = 0 #kg #TODO Values TO_BE_CHECKED

# current_investment = (installedSolarCapacity*solarCost + installedOnWindCapacity*windOnshoreCost
#  + installedOffWindCapacity*windOffshoreCost +  storageCost*storageCapacity)

# %%
def RenGen_MaxOpt(data, capacityFactors):

    #model type: Concrete as the coefficients of the objective function are specified here
    model = pyo.ConcreteModel()


    model.i = pyo.RangeSet(0, len(data)-1)

    #Renewable Generation Variables

    model.solarGen = pyo.Var(domain=pyo.NonNegativeReals, bounds = (0.0, 300e3))
    model.windGen = pyo.Var(domain=pyo.NonNegativeReals, bounds = (0.0, 300e3))

    model.hydrogenSTOR = pyo.Var(domain=pyo.NonNegativeReals, bounds = (hydrogen_operating_Pmin, hydrogen_operating_Pmax))
    model.electrolyzerFlow = pyo.Var(domain=pyo.NonNegativeReals)
    model.hydrogenGen =pyo.Var(domain=pyo.NonNegativeReals, bounds = (0,hydrogen_powerGen_max))
    model.FuelcellFlow = pyo.Var(domain=pyo.NonNegativeReals)
    model.LOH = pyo.Var(domain=pyo.NonNegativeReals, bounds = (0,hydrogen_tank_capacity))
    model.renGen = pyo.Var(model.i, domain=pyo.NonNegativeReals)

    model.batteryCapacity = pyo.Var(domain=pyo.NonNegativeReals)
    model.SOC = pyo.Var(model.i, domain=pyo.NonNegativeReals, bounds = (SOCmin, SOCmax))
    model.charge = pyo.Var(model.i, domain=pyo.NonNegativeReals, bounds = (0.0, storagePower))
    model.discharge = pyo.Var(model.i, domain=pyo.NonNegativeReals, bounds = (0.0, storagePower))
    
    model.conventionalGen = pyo.Var(model.i, domain=pyo.NonNegativeReals)
    model.renShare = pyo.Var(model.i, domain=pyo.NonNegativeReals)

    model.curtailment = pyo.Var(model.i, domain=pyo.NonNegativeReals)

    # Solar and WInd
    def renGen_rule(model, i):
        return model.renGen[i] == (model.solarGen + (weatherData['PV_gen_real[kW]'].iloc[i]/1000) ) * capacityFactors['solar'].iloc[i] \
            + (model.windGen + (weatherData['wind_gen[kW]'].iloc[i]/1000)) * capacityFactors['onshore'].iloc[i] \

    #Battery storage
    def SOC_rule(model, i):
        if i == 0:
            return model.SOC[i] == initialSOC * (storageCapacity + model.batteryCapacity) * (1 - self_discharge_rate) #+ model.charge[i] * chargingEfficiency - model.discharge[i] / dischargingEfficiency
        else:
            return model.SOC[i] == model.SOC[i-1]*(1 - self_discharge_rate) + model.charge[i-1] * chargingEfficiency - model.discharge[i-1] / dischargingEfficiency

    #Hydrogen Gen
    def HydrogenElectrolyzer_rule(model, i):
        return model.hydrogenGen[i] == HHV_H2 * model.electrolyzerFlow[i] / H2_electrolyzer_eff
    #Hydrogen Stor
    def HydrogenFuelcell_rule(model, i):
        return model.hydroSTOR[i] == LHV_H2 * model.FuelcellFlow[i] * H2_fuelcell_eff
    def HydrogenTank_rule(model, i):
        return model.LOH[i] == model.LOH[i-1] + model.electrolyzerFlow[i-1] - model.FuelcellFlow[i-1]

    def energyBalance_rule(model, i):
        return data['demand'].iloc[i] + model.curtailment[i] + model.charge[i] + model.hydroSTOR[i] == model.conventionalGen[i] + model.renGen[i] + model.discharge[i] + model.hydroGEN[i]
    
    
    def renShare_rule(model, i):
        return model.renShare[i] == 1 - model.conventionalGen[i] / data['demand'].iloc[i]
    
    
    def renShareTarget_rule(model):
        return ((renewableShareTarget-0.001), pyo.summation(model.renShare)/len(data), (renewableShareTarget+0.001))
    
    
    def batteryCapacity_rule(model, i):
        return model.SOC[i] <= model.batteryCapacity + storageCapacity
        

    
    model.renGen_rule = pyo.Constraint(model.i, rule=renGen_rule)
    model.SOC_rule = pyo.Constraint(model.i, rule=SOC_rule)
    model.energyBalance_rule = pyo.Constraint(model.i, rule=energyBalance_rule)
    model.renShare_rule = pyo.Constraint(model.i, rule=renShare_rule)
    model.renShareTarget_rule = pyo.Constraint(rule=renShareTarget_rule)
    model.batteryCapacity_rule = pyo.Constraint(model.i, rule = batteryCapacity_rule)
    model.HydrogenElectrolyzer_rule = pyo.Constraint(model.i, rule=HydrogenElectrolyzer_rule)
    model.HydrogenFuelcell_rule = pyo.Constraint(model.i, rule=HydrogenFuelcell_rule)
    model.HydrogenTank_rule = pyo.Constraint(model.i, rule=HydrogenTank_rule)

    def ObjRule(model):
        return model.renShare_rule
     
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

model = RenGen_MaxOpt(data, capacityFactors)

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