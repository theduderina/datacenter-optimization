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
def plot_gendem(GenDem, pv, wind, demand):
    """Plot pv, wind and datacenter-demand."""
    fig, ax = plt.subplots()

    ax.set_title('Renewables Generation and Datacenter Demand')

    ax.plot(GenDem.index, pv, label='PV generation')
    ax.plot(GenDem.index, wind, label='Wind generation')
    ax.plot(GenDem.index, demand, label='datacenter demand')

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power')
    ax.legend()
    ax.minorticks_on()
    ax.set_xlim(0, 8748)

    fig.tight_layout()
    fig.savefig("output/gendem.pdf", transparent=True, bbox_inches="tight")
    fig.show()


@mpltex.acs_decorator
def plot_batt_charge(Batt_charge):
    fig, ax = plt.subplots()

    ax.set_title('batt_charge')

    ax.plot(Batt_charge, marker='o', label='Z')

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power')

    ax.minorticks_on()
    ax.set_xlim(-10, )

    fig.tight_layout()
    #fig.savefig("output/renshare.pdf", transparent=True, bbox_inches="tight")
    fig.show()


@mpltex.acs_decorator
def plot_batt_discharge(Batt_discharge):
    fig, ax = plt.subplots()

    ax.set_title('batt_discharge')

    ax.plot(Batt_discharge, marker='o', label='Z')

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power')

    ax.minorticks_on()
    ax.set_xlim(-10, )

    fig.tight_layout()
    #fig.savefig("output/renshare.pdf", transparent=True, bbox_inches="tight")
    fig.show()


@mpltex.acs_decorator
def plot_gen(GenDem, Prod):
    fig, ax = plt.subplots()

    ax.set_title('Electricity Generation')

    ax.plot((GenDem['demand in kW']) / 1000, label='Demand')  #marker='o'
    #ax.plot(convGen, label='conventional generation', marker='o')
    ax.plot(Prod, label='Prod')

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power')  # in W?
    ax.legend()
    ax.minorticks_on()
    ax.set_xlim(-10, 300)

    fig.tight_layout()
    fig.savefig("output/electricity-generation.pdf",
                transparent=True,
                bbox_inches="tight")
    fig.show()


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


@mpltex.acs_decorator
def plot_prod(Prod, renGen):
    fig, ax = plt.subplots()

    ax.set_title('Production and Renewable Generation')

    ax.plot(Prod, label='Prod')
    ax.plot(renGen, label='renewable generation')  #marker='o

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power')  # in W?
    ax.legend()
    ax.minorticks_on()
    #ax.set_ylim(597000,600100)

    fig.tight_layout()
    fig.savefig("output/prod.pdf", transparent=True, bbox_inches="tight")
    fig.show()


@mpltex.acs_decorator
def plot_loh(LoH):
    fig, ax = plt.subplots()

    ax.set_title('Level of Hydrogen Tank')

    ax.plot(LoH)

    ax.set_xlabel("Hours")
    #ax.set_ylabel('jfg')
    ax.legend()
    ax.minorticks_on()
    ax.set_xlim(-10, )
    #ax.set_ylim(4999,5100)

    fig.tight_layout()
    fig.savefig("output/loh.pdf", transparent=True, bbox_inches="tight")
    fig.show()


@mpltex.acs_decorator
def plot_Electrolyzer(Electrolyzer):
    fig, ax = plt.subplots()

    ax.set_title('Charging H2')

    ax.plot(Electrolyzer)

    ax.set_xlabel("Hours")
    #ax.set_ylabel('jfg')
    ax.legend()
    ax.minorticks_on()
    ax.set_xlim(-10, )
    #ax.set_ylim(49000,51000)

    fig.tight_layout()
    #fig.savefig("output/loh.pdf", transparent=True, bbox_inches="tight")
    fig.show()


@mpltex.acs_decorator
def plot_FuelCell(FuelCell):
    fig, ax = plt.subplots()

    ax.set_title('Discharging H2')

    ax.plot(FuelCell)

    ax.set_xlabel("Hours")
    #ax.set_ylabel('jfg')
    ax.legend()
    ax.minorticks_on()
    ax.set_xlim(-10, )
    #ax.set_ylim(49000,51000)

    fig.tight_layout()
    #fig.savefig("output/loh.pdf", transparent=True, bbox_inches="tight")
    fig.show()


@mpltex.acs_decorator
def plot_Batt(Batt):
    fig, ax = plt.subplots()

    ax.set_title('Battery')

    ax.plot(Batt)

    ax.set_xlabel("Hours")
    #ax.set_ylabel('jfg')
    ax.legend()
    ax.minorticks_on()
    ax.set_xlim(-10, )
    #ax.set_ylim(49000,51000)

    fig.tight_layout()
    #fig.savefig("output/loh.pdf", transparent=True, bbox_inches="tight")
    fig.show()


@mpltex.acs_decorator
def plot_renshare(renShare):
    fig, ax = plt.subplots()

    ax.set_title('Renewable Share')

    ax.plot(renShare, marker='o')

    ax.set_xlabel("Hours")
    ax.set_ylabel('Power')

    ax.minorticks_on()
    ax.set_xlim(-10, )

    fig.tight_layout()
    fig.savefig("output/renshare.pdf", transparent=True, bbox_inches="tight")
    fig.show()
