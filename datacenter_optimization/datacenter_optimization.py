#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
"""
Datacenter optimization.

Student name: Jay Bhavesh Doshi , Anna Lebowsky
Student matriculation number: 4963577 , 5143788
"""
#%%
import time
import os
from datetime import datetime

import numpy as np
import pandas as pd  # import pandas to work with dataframes
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from plots import (
    plot_gendem,
    plot_batt,
    plot_gen,
    plot_prod,
    plot_loh,
    plot_Electrolyzer,
<<<<<<< HEAD
    plot_FuelCell
)
=======
    plot_FuelCell,
    plot_Batt)

>>>>>>> ef090a89b37db6e7f1543ed532f307c41745646c
from utils import INPUT_PATH, utcfromtimestamp

# %% Import weather and demand data

weatherData = pd.read_csv(os.path.join(INPUT_PATH, "weather-data_wind-pv_Freiburg.csv"),
                          index_col=0,
                          parse_dates=True)

# Sum generation of 4 wind turbines
wind_cols = [f"wind_{i}r[kW]" for i in range(1, 5)]
weatherData['wind_gen'] = weatherData[wind_cols].sum(axis=1)

# Create arrays for wind and pv generation and calculate mean hourly values
wind = np.array(weatherData["wind_gen"])[:-1]
pv = np.array(weatherData["PV_gen_real[kW]"])[:-1]

wind = wind.reshape(len(wind) // 4, 4).mean(axis=1)
pv = pv.reshape(len(pv) // 4, 4).mean(axis=1)


[f"{i}{j}" for i in "abc" for j in "12"]

l_unit_name = [f"{i}{j}" for i in "abc" for j in "12"]

for i, unit_name in enumerate(l_unit_name):
    df_temp = pd.read_csv(
        os.path.join(INPUT_PATH, f"hp-s332-{unit_name}.out.gz"),
        names=["datetime", f"power_{unit_name}"],
        converters={
            0: utcfromtimestamp,
            1: float
        },
        sep='\s+'  # seperator between columns is a space
    )
    # Convert NaN to 0
    df_temp[f"power_{unit_name}"] = np.nan_to_num(
        df_temp[f"power_{unit_name}"])

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
df_mean = df_1617.groupby(
    [df_1617.index.month, df_1617.index.day, df_1617.index.hour]).mean()

# Create an array of hourly power values and transform W in kW
demand = np.array(
    df_mean["power"]
) / 1000 * 35  #upscaling factor for demand, probably different scenarios *50 looks good, *10 too low, *100 too high

# Create a dataframe with wind+pv-generation and datacenter-demand
GenDem = pd.DataFrame(np.vstack([pv, wind, demand]).T,
                      columns=["pv in kW", "wind in kW", "demand in kW"])

# create output folder
try:
    os.mkdir("output")
except OSError:
    # Ignore error if folder already exists
    pass

#-------------------- Defining technology sizing, efficiency, and boundary parameters------------------------------------------
n_inverter = 0.8
#Battery
self_discharge_rate = 0.002  # derived from 5% in 24h: https://batteryuniversity.com/article/bu-802b-what-does-elevated-self-discharge-do
storageCapacity = 20  #in MWh (2000GWh for another cse)
storagePower = 20  #in MW (updating w.r.t change in capacity)
chargingEfficiency = 0.82
dischargingEfficiency = 0.92
initialSOC = 0.2 * storageCapacity  #initial State of Charge (ratio from capacity)
SOCmin = 0.1 * storageCapacity  #
SOCmax = 0.9 * storageCapacity  #

#H2_electrolyzer = Hydrogen storage
hydrogen_operating_Pmin = 0  #MW
hydrogen_operating_Pmax = 5  #MW
H2_electrolyzer_eff = 0.6  #TODO Values TO_BE_CHECKED
HHV_H2 = 197  #MWh/kg for 50000Kg H2 found on https://h2tools.org/hyarc/calculator-tools/lower-and-higher-heating-values-fuels

#H2_fuel cell = Hydrogen generation
hydrogen_powerGen_max = 5  #MW
H2_fuelcell_eff = 0.68
LHV_H2 = 167  #MWh/kg for 50000Kg H2 found on https://h2tools.org/hyarc/calculator-tools/lower-and-higher-heating-values-fuels

#Hydrogen tank
hydrogen_tank_capacity = 5000  #kg

# relax factor for achieving 85% of demand
rf = 0.85
#----------------------Initiating for defining time-horizon--------------------------------------------------------------------
# for calculating the time for running the model for specific time-horizon
start_time = time.time()

# Defining the time-horizon for the model in weeks
N_WEEKS = 4
N_HOURS = N_WEEKS * 7 * 24
print(f"Time horizon is {N_WEEKS} weeks ({N_HOURS} hours).")

# ------------------------------Defining Optimization function------------------------------------------------------------------
def RenGen_MaxOpt(GenDem):

    #model type: Concrete as the coefficients of the objective function are specified here
    model = pyo.ConcreteModel()

    #Defining model time-horizon's starting and end-point
    model.i = pyo.RangeSet(0, N_HOURS)

    #Model variables for further constraint definitions: for each time step or for total time horizon

    #Hydrogn electrolyzer and fuel cell for charging and discharging
    model.hydrogenSTOR = pyo.Var(model.i,
                                 domain=pyo.NonNegativeReals,
                                 bounds=(hydrogen_operating_Pmin,
                                         hydrogen_operating_Pmax))
    model.electrolyzerFlow = pyo.Var(model.i, domain=pyo.NonNegativeReals)
    model.hydrogenGEN = pyo.Var(model.i,
                                domain=pyo.NonNegativeReals,
                                bounds=(0, hydrogen_powerGen_max))
    model.FuelcellFlow = pyo.Var(model.i, domain=pyo.NonNegativeReals) #in Kgs
    model.LOH = pyo.Var(
        model.i,
        domain=pyo.NonNegativeReals,
        bounds=(0, hydrogen_tank_capacity))  #level of hydrogen tank


    model.renGen = pyo.Var(model.i, domain=pyo.NonNegativeReals)
    model.Production = pyo.Var(
        model.i, domain=pyo.NonNegativeReals)

    #optimization variable
    model.total = pyo.Var(domain=pyo.Reals)

    #Battery storage variables
    model.batteryCapacity = pyo.Var(domain=pyo.NonNegativeReals)
    model.SOC = pyo.Var(model.i,
                        domain=pyo.NonNegativeReals,
                        bounds=(SOCmin, SOCmax))
    model.charge = pyo.Var(model.i,
                           domain=pyo.NonNegativeReals,
                           bounds=(0.0, storagePower))
    model.discharge = pyo.Var(model.i,
                              domain=pyo.NonNegativeReals,
                              bounds=(0.0, storagePower))

    #Binary variables
    model.x = pyo.Var(model.i, domain=pyo.NonNegativeIntegers, bounds=(0, 1))
    model.y = pyo.Var(model.i, domain=pyo.NonNegativeIntegers, bounds=(0, 1))
    model.z = pyo.Var(model.i, domain=pyo.NonNegativeIntegers, bounds=(0, 1))

    # Solar and Wind Generation
    def renGen_rule(model, i):
        return model.renGen[i] == ((GenDem["pv in kW"].iloc[i] / 1000) +
                                   ((GenDem["wind in kW"].iloc[i] / 1000))
                                   )

    # Battery storage capacity maintainance
    def SOC_rule(model, i):
        if i == 0:
            return model.SOC[i] == initialSOC * (
                1 - self_discharge_rate
            )
        else:
            return model.SOC[i] == model.SOC[i - 1] * (
                1 - self_discharge_rate
            ) + model.charge[i - 1] * chargingEfficiency - model.discharge[
                i - 1] / dischargingEfficiency

    # Hydrogen Storage
    def HydrogenElectrolyzer_rule(model, i):
        return model.hydrogenSTOR[
            i] == HHV_H2 * model.electrolyzerFlow[i] / H2_electrolyzer_eff

    # Hydrogen Generation
    def HydrogenFuelcell_rule(model, i):
        return model.hydrogenGEN[
            i] == LHV_H2 * model.FuelcellFlow[i] * H2_fuelcell_eff

    # Hydrogen tank capacity maintainance
    def HydrogenTank_rule(model, i):
        if i == 0:
            return model.LOH[i] == hydrogen_tank_capacity
        else:
            return model.LOH[i] == model.LOH[i - 1] + model.electrolyzerFlow[
                i - 1] - model.FuelcellFlow[i - 1]

    # Cummulating renewable generation with storage discharging and charging
    def Production_rule(model, i):
        return model.Production[i] <= model.renGen[i] + (
            model.hydrogenGEN[i] + model.discharge[i]) * n_inverter - (
                model.hydrogenSTOR[i] + model.charge[i]) * n_inverter

    # Achieving atleast predefined relax facor
    def relax_factor_rule(model, i):
        return model.Production[i] >= rf * GenDem["demand in kW"].iloc[i] / 1000

    # Objective function to maintain production with Level of hydrogen
    def Hourly_power_production_rule(model):
        for i in model.i:
            return model.total == model.Production[i] + model.LOH[i]

    # Binary logics for mutually exclusive events: Battery and hydrogen discharge should
    # only suffice demand and not charge each other
    def Mut_excl_Battcharge_rule(model, i):
        return model.charge[i] <= model.x[i] * storagePower

    def Mut_excl_Battdischarge_rule(model, i):
        return model.discharge[i] <= (1 - model.x[i]) * storagePower

    def Mut_excl_H2_storagemax_rule(model, i):
        return model.hydrogenSTOR[i] <= model.y[i] * hydrogen_operating_Pmax

    def Mut_excl_H2_storagemin_rule(model, i):
        return model.hydrogenSTOR[i] >= model.y[i] * hydrogen_operating_Pmin

    def Mut_excl_H2_Genmax_rule(model, i):
        return model.hydrogenGEN[i] <= (1 -
                                        model.y[i]) * hydrogen_operating_Pmax

    def Mut_excl_H2_Genmin_rule(model, i):
        return model.hydrogenGEN[i] >= (1 -
                                        model.y[i]) * hydrogen_operating_Pmin

    def Mut_excl_H2_eleflow_rule(model, i):
        return model.electrolyzerFlow[i] <= model.z[i] * hydrogen_tank_capacity

    def Mut_excl_H2_fuelcellflow_rule(model, i):
        return model.FuelcellFlow[i] <= (1 -
                                         model.z[i]) * hydrogen_tank_capacity

    def Mut_excl_Binary1_rule(model, i):
        return model.x[i] + model.y[i] == 1

    def Mut_excl_Binary2_rule(model, i):
        return model.x[i] + model.z[i] == 1

    # CONVERTING DEFINED FUNCTIONS INTO CONSTRAINTS
    model.renGen_rule = pyo.Constraint(model.i, rule=renGen_rule)
    model.Production_rule = pyo.Constraint(model.i, rule=Production_rule)
    model.SOC_rule = pyo.Constraint(model.i, rule=SOC_rule)
    model.HydrogenElectrolyzer_rule = pyo.Constraint(model.i,
                                                rule=HydrogenElectrolyzer_rule)
    model.HydrogenFuelcell_rule = pyo.Constraint(model.i,
                                                 rule=HydrogenFuelcell_rule)
    model.HydrogenTank_rule = pyo.Constraint(model.i, rule=HydrogenTank_rule)

    model.relax_factor_rule = pyo.Constraint(model.i, rule=relax_factor_rule)
    model.Hourly_power_production_rule = pyo.Constraint(
        model.i, rule=Hourly_power_production_rule)

    model.Mut_excl_Battcharge_rule = pyo.Constraint(
        model.i, rule=Mut_excl_Battcharge_rule)
    model.Mut_excl_Battdischarge_rule = pyo.Constraint(
        model.i, rule=Mut_excl_Battdischarge_rule)
    model.Mut_excl_H2_storagemax_rule = pyo.Constraint(
        model.i, rule=Mut_excl_H2_storagemax_rule)
    model.Mut_excl_H2_storagemin_rule = pyo.Constraint(
        model.i, rule=Mut_excl_H2_storagemin_rule)
    model.Mut_excl_H2_eleflow_rule = pyo.Constraint(
        model.i, rule=Mut_excl_H2_eleflow_rule)
    model.Mut_excl_H2_fuelcellflow_rule = pyo.Constraint(
        model.i, rule=Mut_excl_H2_fuelcellflow_rule)
    model.Mut_excl_Binary1_rule = pyo.Constraint(model.i,
                                                 rule=Mut_excl_Binary1_rule)
    model.Mut_excl_Binary2_rule = pyo.Constraint(model.i,
                                                 rule=Mut_excl_Binary2_rule)
    model.Mut_excl_H2_Genmin_rule = pyo.Constraint(
        model.i, rule=Mut_excl_H2_Genmin_rule)
    model.Mut_excl_H2_Genmax_rule = pyo.Constraint(
        model.i, rule=Mut_excl_H2_Genmax_rule)

    # Objective function of the maximization problem
    def ObjRule(model):
        return model.total

    model.obj = pyo.Objective(rule=ObjRule, sense=pyo.maximize)

    opt = SolverFactory("glpk")

    opt.solve(model)

    return model


# ------------------------------Running Optimization function with PV, Wind and demand data----------------------------------------
model = RenGen_MaxOpt(GenDem)

# Printing model run time
print("--%s mins--" % (time.time() - start_time))


# ------------------------------Extracting data from the model simulation for further analysis----------------------------------------
def get_values(model):
    renGen = []
    Prod = []
    LoH = []
    Batt = []
    Z = []
    X = []
    Y = []
    Electrolyzer = []
    Batt_charge = []
    Batt_discharge = []
    FuelCell = []
    for i in range(N_HOURS):
        Prod.append(model.Production[i].value)
        LoH.append(model.LOH[i].value)
        Batt.append(model.SOC[i].value)
        renGen.append(model.renGen[i].value)
        Z.append(model.z[i].value)
        Y.append(model.y[i].value)
        X.append(model.x[i].value)
        Electrolyzer.append(model.hydrogenSTOR[i].value)
        FuelCell.append(model.hydrogenGEN[i].value)
        Batt_charge.append(model.charge[i].value)
        Batt_discharge.append(model.discharge[i].value)
    return renGen, Prod, LoH, Batt, Z, Y, X, Electrolyzer, FuelCell, Batt_charge, Batt_discharge

renGen, Prod, LoH, Batt, Z, Y, X, Electrolyzer, FuelCell, Batt_charge, Batt_discharge = get_values(model)

# ------------------------------Calculating the Metrics-----------------------------------------------------------------------------

demand = np.array(demand[:N_HOURS]) /1000

# Loss of Power Supply Probability (LPSP)
lpsp = np.sum(np.array(Prod) < demand) / N_HOURS
print(f"lpsp = {lpsp}")

# Level of Autonomy (LA)
la = np.sum(np.array(renGen) >= demand) / N_HOURS
print(f"la = {la}")

# Unused Renewable Energy (URE)
delta = Prod - demand
greater_zero = delta > 0
ure = np.sum(delta * greater_zero) / N_HOURS
print(f"ure = {ure}")

# Percentage of the Energy Produced to demand (PEP)
pep = (np.sum(np.minimum(Prod, demand))) / np.sum(demand)
print(f"pep = {pep}")

# Plotting results
plot_gendem(pv, wind, demand)
plot_gen(GenDem, Prod)
plot_FuelCell(FuelCell)
plot_Electrolyzer(Electrolyzer)
plot_prod(Prod, renGen)
plot_loh(LoH)
plot_batt(Batt, Batt_charge ,Batt_discharge)
