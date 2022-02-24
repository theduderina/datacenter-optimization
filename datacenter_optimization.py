# -*- coding: utf-8 -*-
"""

Student name: Jay Bhavesh Doshi , Anna Lebowsky
Student matriculation number: 4963577 , 5143788
              

"""
# %%
# Aim: To find the optimized capacity of solar, on-shore and off-shore wind
# to achieve the maximum renewable energy share. 
#%%
import numpy as np
import pandas as pd #import pandas to work with dataframes
from datetime import datetime
import mpltex # for nice plots
import math as m
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import time
import matplotlib.pyplot as plt
# %% Import weather and demand data

weatherData = pd.read_csv('input/weather-data_wind-pv_Freiburg.csv', index_col=0, parse_dates = True)

# Sum generation of 4 wind turbines
wind_cols = [f"wind_{i}r[kW]" for i in range(1,5)]
weatherData['wind_gen'] = weatherData[wind_cols].sum(axis=1)

# Create arrays for wind and pv generation and calculate mean hourly values
wind = np.array(weatherData["wind_gen"])[:-1]
pv = np.array(weatherData["PV_gen_real[kW]"])[:-1]

wind = wind.reshape(len(wind) // 4, 4).mean(axis=1)
pv = pv.reshape(len(pv) // 4, 4).mean(axis=1)

def utcfromtimestamp(timestamp):
    """
    Construct a naive UTC datetime from a POSIX timestamp.
    
    Same as `datetime.utcfromtimestamp` but can also handle str as
    input.
    """
    
    return datetime.utcfromtimestamp(int(timestamp))

[f"{i}{j}" for i in "abc" for j in "12"]

l_unit_name = [f"{i}{j}" for i in "abc" for j in "12"]

for i, unit_name in enumerate(l_unit_name):
    df_temp = pd.read_csv(f"input/hp-s332-{unit_name}.out.gz",
                          names=["datetime", f"power_{unit_name}"],
                          converters={0: utcfromtimestamp,
                                      1: float},
                          sep='\s+'  # seperator between columns is a space
                          )
    # Convert NaN to 0
    df_temp[f"power_{unit_name}"] = np.nan_to_num(df_temp[f"power_{unit_name}"])
    
    # Merge into one dataframe
    if i == 0:
        # If the loop just started the first unit `df` does not exists so we 
        # create this one here
        df = df_temp
    else:
        df = df.merge(df_temp)

# Sum power of all units
power_cols = [f"power_{i}" for i in l_unit_name]
df['power'] = df[power_cols].sum(axis=1)

# Resample data for every hour
df.index = df['datetime']
df = df.resample('H').mean()

start_date = datetime.fromisoformat("2016-01-01")
end_date = datetime.fromisoformat("2018-01-01")

df_1617 = df.loc[(df.index >= start_date) & (df.index < end_date)]

# Calculate hourly mean values out of 2016 & 2017
df_mean = df_1617.groupby([df_1617.index.month, df_1617.index.day, df_1617.index.hour]).mean()

# Create an array of hourly power values and transform W in kW
demand = np.array(df_mean["power"]) / 1000 * 35  #upscaling factor for demand, probably different scenarios *50 looks good, *10 too low, *100 too high

# Plotting power demand from datacenter
# plt.plot(demand)
# plt.ylabel("Power in kW")
# plt.xlabel("Hours")
# plt.show()

# Create a dataframe with wind+pv-generation and datacenter-demand
GenDem = pd.DataFrame(np.vstack([pv, wind, demand]).T, columns=["pv in kW",
                                                               "wind in kW",
                                                               "demand in kW"])

#Plot pv, wind and datacenter-demand
#@mpltex.acs_decorator
def plot_gendem():
    fig, ax = plt.subplots()

    ax.set_title('Renewables Generation and Datacenter Demand')

    ax.plot(GenDem.index, pv, label='PV generation')
    ax.plot(GenDem.index, wind, label='Wind generation')
    ax.plot(GenDem.index, demand, label='datacenter demand')

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power')
    ax.legend()
    ax.minorticks_on()
    ax.set_xlim(0,8748)

    fig.tight_layout()
    fig.savefig("output/gendem.pdf", transparent=True, bbox_inches="tight")
    fig.show()

plot_gendem()

#%%

# installedSolarCapacity = 53110 #MW
# installedWindCapacity = 54490 #MW

#renewableShareTarget = 1 #0.6 for another case

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
n_inverter = 0.8
#Battery
self_discharge_rate = 0.002 # derived from 5% in 24h: https://batteryuniversity.com/article/bu-802b-what-does-elevated-self-discharge-do
storageCapacity = 20 #in MWh (2000GWh for another cse)
storagePower = 20 #in MW (updating w.r.t change in capacity)
chargingEfficiency = 0.82
dischargingEfficiency = 0.92
initialSOC = 0.2*storageCapacity #initial State of Charge (ratio from capacity)
SOCmin = 0.1*storageCapacity #
SOCmax = 0.9*storageCapacity #

#H2_electrolyzer = Hydrogen storage
hydrogen_operating_Pmin = 0 #MW
hydrogen_operating_Pmax = 5 #MW
H2_electrolyzer_eff = 0.6 #TODO Values TO_BE_CHECKED
HHV_H2 = 197 #MWh/kg for 50000Kg H2 found on https://h2tools.org/hyarc/calculator-tools/lower-and-higher-heating-values-fuels


#H2_fuel cell = Hydrogen generation
hydrogen_powerGen_max = 5 #MW
H2_fuelcell_eff = 0.68 #TODO Values TO_BE_CHECKED
LHV_H2 = 167  #MWh/kg for 50000Kg H2 found on https://h2tools.org/hyarc/calculator-tools/lower-and-higher-heating-values-fuels

rf = 0.85
#Hydrogen_tank
hydrogen_tank_capacity = 5000 #kg #TODO Values TO_BE_CHECKED

start_time = time.time()
# %%
def RenGen_MaxOpt(GenDem):
#, pv, wind, demand

    #model type: Concrete as the coefficients of the objective function are specified here
    model = pyo.ConcreteModel()

    # Defining the time-horizon for the model
    model.i = pyo.RangeSet(0, 8748)

 #Model variables for further constraint definitions: for each time step or for total time horizon

#% ------------------------------------------------

        #Renewable Generation Variables
    #model.solarGen = pyo.Var(domain=pyo.NonNegativeReals, bounds = (0.0, 300e3))
    #model.windGen = pyo.Var(domain=pyo.NonNegativeReals, bounds = (0.0, 300e3))

         #Hydrogn electrolyzer and fuel cell for charging and discharging
    model.hydrogenSTOR = pyo.Var(model.i,domain=pyo.NonNegativeReals, bounds=(hydrogen_operating_Pmin, hydrogen_operating_Pmax))
    # model.lammbda = pyo.Var( domain=pyo.NonNegativeReals, bounds=(0,1))
    model.electrolyzerFlow = pyo.Var(model.i,domain=pyo.NonNegativeReals)
    model.hydrogenGEN =pyo.Var(model.i,domain=pyo.NonNegativeReals, bounds = (0,hydrogen_powerGen_max))
    model.FuelcellFlow = pyo.Var(model.i,domain=pyo.NonNegativeReals)
    model.LOH = pyo.Var(model.i,domain=pyo.NonNegativeReals, bounds = (0,hydrogen_tank_capacity)) #level of hydrogen tank

    model.renGen = pyo.Var(model.i, domain=pyo.NonNegativeReals)
    model.Production = pyo.Var(model.i, domain=pyo.NonNegativeReals) #Reals or NonNegative ????
    model.total = pyo.Var(domain=pyo.Reals)

    model.batteryCapacity = pyo.Var(domain=pyo.NonNegativeReals)
    model.SOC = pyo.Var(model.i, domain=pyo.NonNegativeReals, bounds = (SOCmin, SOCmax))
    model.charge = pyo.Var(model.i, domain=pyo.NonNegativeReals, bounds = (0.0, storagePower))
    model.discharge = pyo.Var(model.i, domain=pyo.NonNegativeReals, bounds = (0.0, storagePower))

    model.conventionalGen = pyo.Var(model.i, domain=pyo.NonNegativeReals)
    # model.rf = pyo.Var(domain=pyo.NonNegativeReals, bounds=(0,1))
    model.renShare = pyo.Var(model.i, domain=pyo.NonNegativeReals)


    model.x = pyo.Var(model.i, domain=pyo.NonNegativeIntegers, bounds = (0,1))
    model.y = pyo.Var(model.i, domain=pyo.NonNegativeIntegers, bounds = (0,1))
    model.z = pyo.Var(model.i, domain=pyo.NonNegativeIntegers, bounds = (0,1))
    # Solar and WInd
    def renGen_rule(model, i):
        return model.renGen[i] == ((GenDem["pv in kW"].iloc[i]/1000) + ((GenDem["wind in kW"].iloc[i]/1000))) #* capacityFactors['solar'].iloc[i] \

    #Battery storage
    def SOC_rule(model, i):
        if i == 0:
            return model.SOC[i] == initialSOC * (1 - self_discharge_rate) #+ model.charge[i] * chargingEfficiency - model.discharge[i] / dischargingEfficiency
        else:
            return model.SOC[i] == model.SOC[i-1]*(1 - self_discharge_rate) + model.charge[i-1] * chargingEfficiency - model.discharge[i-1] / dischargingEfficiency

    #Hydrogen Stor
    def HydrogenElectrolyzer_rule(model, i):
        return model.hydrogenSTOR[i] == HHV_H2 * model.electrolyzerFlow[i] / H2_electrolyzer_eff
    #Hydrogen Gen
    def HydrogenFuelcell_rule(model, i):
        return model.hydrogenGEN[i] == LHV_H2 * model.FuelcellFlow[i] * H2_fuelcell_eff
    def HydrogenTank_rule(model, i):
        if i == 0:
            return model.LOH[i] == hydrogen_tank_capacity
        else:
            return model.LOH[i] == model.LOH[i-1] + model.electrolyzerFlow[i-1] - model.FuelcellFlow[i-1]

    def Production_rule(model, i):
        return model.Production[i] <= model.renGen[i] + (model.hydrogenGEN[i] + model.discharge[i])*n_inverter - (model.hydrogenSTOR[i] + model.charge[i])*n_inverter

    def renShare_rule(model, i):
         return model.renShare[i] == model.Production[i] / (GenDem["demand in kW"].iloc[i]/1000)

    def relax_factor_rule(model, i):
        return model.Production[i] >= rf * GenDem["demand in kW"].iloc[i]/1000

    def Hourly_power_production_rule(model): #TODO: Check the summation for whole horizon(with or without i)
        for i in model.i:
            return model.total == model.Production[i] + model.LOH[i] # for i in model.i)  (model.lammbda)*


    def Mut_excl_Battcharge_rule(model,i):
        return model.charge[i] <= model.x[i] * storagePower

    def Mut_excl_Battdischarge_rule(model,i):
        return model.discharge[i] <= (1-model.x[i]) * storagePower

    def Mut_excl_H2_storagemax_rule(model,i):
        return model.hydrogenSTOR[i] <= model.y[i] * hydrogen_operating_Pmax

    def Mut_excl_H2_storagemin_rule(model,i):
        return model.hydrogenSTOR[i] >= model.y[i] * hydrogen_operating_Pmin

    def Mut_excl_H2_Genmax_rule(model,i):
        return model.hydrogenGEN[i] <= (1-model.y[i]) * hydrogen_operating_Pmax

    def Mut_excl_H2_Genmin_rule(model,i):
        return model.hydrogenGEN[i] >= (1-model.y[i]) * hydrogen_operating_Pmin

    def Mut_excl_H2_eleflow_rule(model,i):
        return model.electrolyzerFlow[i] <= model.z[i] * hydrogen_tank_capacity

    def Mut_excl_H2_fuelcellflow_rule(model,i):
        return model.FuelcellFlow[i] <= (1-model.z[i]) * hydrogen_tank_capacity

    # def mut_h2tobatt_rule(model,i):
    #     if i == 0:
    #         return
    #     if i != 0:
    #         if model.z[i] == 0 or model.y[i] == 0:
    #             return model.x[i] == 0
    #         if model.x[i] == 0:
    #             return model.y[i] == 0

    def Mut_excl_Binary1_rule(model,i):
        return model.x[i] + model.y[i] == 1

    def Mut_excl_Binary2_rule(model,i):
        return model.x[i] + model.z[i] == 1

    # def renShareTarget_rule(model):
    #     return ((renewableShareTarget-0.001), pyo.summation(model.renShare)/len(data), (renewableShareTarget+0.001))

    # def batteryCapacity_rule(model, i):
    #     return model.SOC[i] <= model.batteryCapacity + storageCapacity




    #model.energyBalance_rule = pyo.Constraint(model.i, rule=energyBalance_rule)
    model.renGen_rule = pyo.Constraint(model.i, rule=renGen_rule)
    model.Production_rule = pyo.Constraint(model.i, rule=Production_rule)
    model.SOC_rule = pyo.Constraint(model.i, rule=SOC_rule)
    model.renShare_rule = pyo.Constraint(model.i, rule=renShare_rule)
    # model.renShareTarget_rule = pyo.Constraint(rule=renShareTarget_rule)
    #model.batteryCapacity_rule = pyo.Constraint(model.i, rule = batteryCapacity_rule)
    model.HydrogenElectrolyzer_rule = pyo.Constraint(model.i, rule=HydrogenElectrolyzer_rule)
    model.HydrogenFuelcell_rule = pyo.Constraint(model.i, rule=HydrogenFuelcell_rule)
    model.HydrogenTank_rule = pyo.Constraint(model.i, rule=HydrogenTank_rule)

    model.relax_factor_rule = pyo.Constraint(model.i, rule=relax_factor_rule)
    model.Hourly_power_production_rule = pyo.Constraint(model.i, rule=Hourly_power_production_rule)

    model.Mut_excl_Battcharge_rule = pyo.Constraint(model.i, rule=Mut_excl_Battcharge_rule)
    model.Mut_excl_Battdischarge_rule = pyo.Constraint(model.i, rule=Mut_excl_Battdischarge_rule)
    model.Mut_excl_H2_storagemax_rule = pyo.Constraint(model.i, rule=Mut_excl_H2_storagemax_rule)
    model.Mut_excl_H2_storagemin_rule = pyo.Constraint(model.i, rule=Mut_excl_H2_storagemin_rule)
    model.Mut_excl_H2_eleflow_rule = pyo.Constraint(model.i, rule=Mut_excl_H2_eleflow_rule)
    model.Mut_excl_H2_fuelcellflow_rule = pyo.Constraint(model.i, rule=Mut_excl_H2_fuelcellflow_rule)
    model.Mut_excl_Binary1_rule= pyo.Constraint(model.i, rule=Mut_excl_Binary1_rule)
    model.Mut_excl_Binary2_rule= pyo.Constraint(model.i, rule=Mut_excl_Binary2_rule)
    model.Mut_excl_H2_Genmin_rule = pyo.Constraint(model.i, rule=Mut_excl_H2_Genmin_rule)
    model.Mut_excl_H2_Genmax_rule = pyo.Constraint(model.i, rule=Mut_excl_H2_Genmax_rule)
    # model.mut_h2tobatt_rule = pyo.Constraint(model.i, rule=mut_h2tobatt_rule)

    def ObjRule(model):
        return model.total

    model.obj = pyo.Objective(rule=ObjRule, sense=pyo.maximize)

    opt = SolverFactory("glpk")


    opt.solve(model)

    return model

model = RenGen_MaxOpt(GenDem)

print("--%s mins--"%(time.time()-start_time))
# %%

def get_values(model):
    renShare = []
    convGen = []
    # curtailed = []
    renGen = []
    Prod = []
    LoH = []
    Batt = []
    Z = []
    X =[]
    Y= []
    Electrolyzer = []
    Batt_charge = []
    Batt_discharge = []
    FuelCell = []
    for i in range(len(GenDem)):
        renShare.append(model.renShare[i].value)
        Prod.append(model.Production[i].value)
        LoH.append(model.LOH[i].value)
        Batt.append(model.SOC[i].value)
        convGen.append(model.conventionalGen[i].value)
        # curtailed.append(model.curtailment[i].value)
        renGen.append(model.renGen[i].value)
        Z.append(model.z[i].value)
        Y.append(model.y[i].value)
        X.append(model.x[i].value)
        Electrolyzer.append(model.hydrogenSTOR[i].value)
        FuelCell.append(model.hydrogenGEN[i].value)
        Batt_charge.append(model.charge[i].value)
        Batt_discharge.append(model.discharge[i].value)
    return renShare, convGen, renGen, Prod, LoH, Batt, Z, Y, X, Electrolyzer, FuelCell, Batt_charge, Batt_discharge


renShare, convGen, renGen, Prod, LoH, Batt, Z, Y, X, Electrolyzer, FuelCell, Batt_charge, Batt_discharge = get_values(model)

#%% Plotting renShare, convGen, curtailed, renGen, Prod, LoH, Batt

# @mpltex.acs_decorator
def plot_batt_charge():
    fig, ax = plt.subplots()

    ax.set_title('batt_charge')

    ax.plot(Batt_charge, marker='o', label= 'Z')

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power')

    ax.minorticks_on()
    ax.set_xlim(-10,)

    fig.tight_layout()
    #fig.savefig("output/renshare.pdf", transparent=True, bbox_inches="tight")
    fig.show()

plot_batt_charge()

def plot_batt_discharge():
    fig, ax = plt.subplots()

    ax.set_title('batt_discharge')

    ax.plot(Batt_discharge, marker='o', label= 'Z')

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power')

    ax.minorticks_on()
    ax.set_xlim(-10,)

    fig.tight_layout()
    #fig.savefig("output/renshare.pdf", transparent=True, bbox_inches="tight")
    fig.show()

plot_batt_discharge()

# @mpltex.acs_decorator
def plot_gen():
    fig, ax = plt.subplots()

    ax.set_title('Electricity Generation')

    ax.plot((GenDem['demand in kW'])/1000, label='Demand') #marker='o'
    #ax.plot(convGen, label='conventional generation', marker='o')
    ax.plot(Prod, label='Prod')

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power') # in W?
    ax.legend()
    ax.minorticks_on()
    ax.set_xlim(-10,300)

    fig.tight_layout()
    fig.savefig("output/electricity-generation.pdf", transparent=True, bbox_inches="tight")
    fig.show()

plot_gen()

# @mpltex.acs_decorator
# def plot_curt():
#     fig, ax = plt.subplots()
#
#     ax.set_title('Curtailed')
#
#     ax.plot(curtailed)
#
#     ax.set_xlabel("Hours")
#     ax.set_ylabel('Power') # in W?
#     ax.legend()
#     ax.minorticks_on()
#     ax.set_xlim(-10,)
#     ax.set_ylim(597000,600100)
#
#     fig.tight_layout()
#     fig.savefig("output/curtailed.pdf", transparent=True, bbox_inches="tight")
#     fig.show()
#
# plot_curt()

# @mpltex.acs_decorator
def plot_prod():
    fig, ax = plt.subplots()

    ax.set_title('Production and Renewable Generation')

    ax.plot(Prod, label='Prod')
    ax.plot(renGen, label='renewable generation') #marker='o

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power') # in W?
    ax.legend()
    ax.minorticks_on()
    #ax.set_ylim(597000,600100)

    fig.tight_layout()
    fig.savefig("output/prod.pdf", transparent=True, bbox_inches="tight")
    fig.show()

plot_prod()

# @mpltex.acs_decorator
def plot_loh():
    fig, ax = plt.subplots()

    ax.set_title('Level of Hydrogen Tank')

    ax.plot(LoH)

    ax.set_xlabel("Hours")
    #ax.set_ylabel('jfg') 
    ax.legend()
    ax.minorticks_on()
    ax.set_xlim(-10,)
    #ax.set_ylim(4999,5100)

    fig.tight_layout()
    fig.savefig("output/loh.pdf", transparent=True, bbox_inches="tight")
    fig.show()

plot_loh()

def plot_Electrolyzer():
    fig, ax = plt.subplots()

    ax.set_title('Charging H2')

    ax.plot(Electrolyzer)

    ax.set_xlabel("Hours")
    #ax.set_ylabel('jfg')
    ax.legend()
    ax.minorticks_on()
    ax.set_xlim(-10,)
    #ax.set_ylim(49000,51000)

    fig.tight_layout()
    #fig.savefig("output/loh.pdf", transparent=True, bbox_inches="tight")
    fig.show()

plot_Electrolyzer()

def plot_FuelCell():
    fig, ax = plt.subplots()

    ax.set_title('Discharging H2')

    ax.plot(FuelCell)

    ax.set_xlabel("Hours")
    #ax.set_ylabel('jfg')
    ax.legend()
    ax.minorticks_on()
    ax.set_xlim(-10,)
    #ax.set_ylim(49000,51000)

    fig.tight_layout()
    #fig.savefig("output/loh.pdf", transparent=True, bbox_inches="tight")
    fig.show()

plot_FuelCell()

def plot_Batt():
    fig, ax = plt.subplots()

    ax.set_title('Battery')

    ax.plot(Batt)

    ax.set_xlabel("Hours")
    #ax.set_ylabel('jfg')
    ax.legend()
    ax.minorticks_on()
    ax.set_xlim(-10,)
    #ax.set_ylim(49000,51000)

    fig.tight_layout()
    #fig.savefig("output/loh.pdf", transparent=True, bbox_inches="tight")
    fig.show()

plot_Batt()

# def Mut_excl_Battcharge_rule(model,i):
#         return model.charge[i] <= model.x[i] * storagePower
#
#     def Mut_excl_Battdischarge_rule(model,i):
#         return model.discharge[i] <= model.y[i] * storagePower
#
#     def Mut_excl_H2_storagemax_rule(model,i):
#         return model.hydrogenSTOR[i] <= model.x[i] * hydrogen_operating_Pmax
#
#     def Mut_excl_H2_storagemin_rule(model,i):
#         return model.hydrogenSTOR[i] >= model.x[i] * hydrogen_operating_Pmin
#
#     def Mut_excl_H2_Genmax_rule(model,i):
#         return model.hydrogenGEN[i] <= model.y[i] * hydrogen_operating_Pmax
#
#     def Mut_excl_H2_Genmin_rule(model,i):
#         return model.hydrogenGEN[i] >= model.y[i] * hydrogen_operating_Pmin
#
#     def Mut_excl_H2_eleflow_rule(model,i):
#         return model.electrolyzerFlow[i] <= model.x[i] * hydrogen_tank_capacity
#
#     def Mut_excl_H2_fuelcellflow_rule(model,i):
#         return model.FuelcellFlow[i] <= model.y[i] * hydrogen_tank_capacity
#
#     def Mut_excl_Binary1_rule(model,i):
#         return model.x[i] + model.y[i] == 1

def plot_renshare():
    fig, ax = plt.subplots()

    ax.set_title('Renewable Share')

    ax.plot(renShare, marker='o')

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power')

    ax.minorticks_on()
    ax.set_xlim(-10,)

    fig.tight_layout()
    fig.savefig("output/renshare.pdf", transparent=True, bbox_inches="tight")
    fig.show()

plot_renshare()
