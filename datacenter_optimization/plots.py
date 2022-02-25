#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
"""
Plot functions.

Student name: Jay Bhavesh Doshi , Anna Lebowsky
Student matriculation number: 4963577 , 5143788
"""
import mpltex  # for nice plots
import matplotlib.pyplot as plt


@mpltex.acs_decorator
def plot_gendem(pv, wind, demand):
    """Plot pv, wind and datacenter-demand."""
    fig, ax = plt.subplots()

    ax.set_title('Renewables Generation and Datacenter Demand')

    ax.plot(pv, label='PV generation')
    ax.plot(wind, label='Wind generation')
    ax.plot(demand, label='datacenter demand')

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power[kW]')
    ax.legend()
    ax.minorticks_on()
    ax.set_xlim(0, 8748)

    fig.tight_layout()
    fig.savefig("output/gendem.pdf", transparent=True, bbox_inches="tight")
    fig.show()


@mpltex.acs_decorator
def plot_batt(Batt_charge, Batt_discharge):
    fig, ax = plt.subplots()

    ax.set_title('Battery Charging and Discharging')

    ax.plot(Batt_charge, label='charging')
    ax.plot(Batt_discharge, label='discharging')

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power in MW')
    ax.legend()

    ax.minorticks_on()
    ax.set_xlim(-10, )

    fig.tight_layout()
    fig.savefig("output/batt.pdf", transparent=True, bbox_inches="tight")
    fig.show()


@mpltex.acs_decorator
def plot_batt_discharge(Batt_discharge):
    fig, ax = plt.subplots()

    ax.set_title('Battery Discharging')

    ax.plot(Batt_discharge, marker='o', label='Z')

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power[MW]')

    ax.minorticks_on()
    ax.set_xlim(-10, )

    fig.tight_layout()
    fig.savefig("output/batt_discharge.pdf", transparent=True, bbox_inches="tight")
    fig.show()


@mpltex.acs_decorator
def plot_gen(GenDem, Prod):
    fig, ax = plt.subplots()

    ax.set_title('Electricity Generation v/s Demand')

    ax.plot((GenDem['demand in kW']) / 1000, label='Demand')
    ax.plot(Prod, label='Prod')

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power[MW]')
    ax.legend()
    ax.minorticks_on()
    ax.set_xlim(-10, 300)

    fig.tight_layout()
    fig.savefig("output/electricity-generation.pdf",
                transparent=True,
                bbox_inches="tight")
    fig.show()



@mpltex.acs_decorator
def plot_prod(Prod, renGen):
    fig, ax = plt.subplots()

    ax.set_title('Production and Renewable Generation')

    ax.plot(Prod, label='Prod')
    ax.plot(renGen, label='renewable generation')

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power[MW]')
    ax.legend()
    ax.minorticks_on()

    fig.tight_layout()
    fig.savefig("output/prod.pdf", transparent=True, bbox_inches="tight")
    fig.show()


@mpltex.acs_decorator
def plot_loh(LoH):
    fig, ax = plt.subplots()

    ax.set_title('Level of Hydrogen Tank')

    ax.plot(LoH)

    ax.set_xlabel("Hours")
    ax.set_ylabel('Level in Kgs')
    ax.legend()
    ax.minorticks_on()
    ax.set_xlim(-10, )

    fig.tight_layout()
    fig.savefig("output/loh.pdf", transparent=True, bbox_inches="tight")
    fig.show()


@mpltex.acs_decorator
def plot_Electrolyzer(Electrolyzer):
    fig, ax = plt.subplots()

    ax.set_title('Charging H2')

    ax.plot(Electrolyzer)

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power[MW]')
    ax.legend()
    ax.minorticks_on()
    ax.set_xlim(-10, )


    fig.tight_layout()
    fig.savefig("output/electrolyzer.pdf", transparent=True, bbox_inches="tight")
    fig.show()


@mpltex.acs_decorator
def plot_FuelCell(FuelCell):
    fig, ax = plt.subplots()

    ax.set_title('Discharging H2')

    ax.plot(FuelCell)

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power[MW]')
    ax.legend()
    ax.minorticks_on()
    ax.set_xlim(-10, )

    fig.tight_layout()
    fig.savefig("output/fuel_cell.pdf", transparent=True, bbox_inches="tight")
    fig.show()



