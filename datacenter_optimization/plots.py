#!/usr/bin/env python3
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
"""
Plot functions.

Student name: Jay Bhavesh Doshi , Anna Lebowsky
Student matriculation number: 4963577 , 5143788
"""
import numpy as np
import mpltex  # for nice plots
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

from mpltex.acs import _height, width_double_column


@mpltex.acs_decorator
def plot_results(Prod, demand, renGen, Electrolyzer, FuelCell, Batt_charge,
                 Batt_discharge, LoH, Batt):

    fig, ax = plt.subplots(6,
                           sharex=True,
                           figsize=[width_double_column, 4 * _height])

    ax_prod = ax[0]
    ax_rengen = ax[1]
    ax_bcharge = ax[2]
    ax_soc = ax[3]
    ax_h2 = ax[4]
    ax_loh = ax[5]

    # Production and Demand
    ax_prod.plot(Prod, label='Production')
    ax_prod.plot(demand, label='Datacenter Demand')
    ax_prod.set_ylabel('Power in MW')

    # Rengen
    ax_rengen.plot(renGen, label='Reneweable Energy Generation')
    ax_rengen.plot(demand, label='Datacenter Demand')
    ax_rengen.set_ylabel('Power in MW')

    # Battery Charging and Discharging
    ax_bcharge.bar(np.arange(len(Batt_charge)),
                   Batt_charge,
                   label='Battery Charging')
    ax_bcharge.bar(np.arange(len(Batt_discharge)),
                   Batt_discharge,
                   label='Battery Discharging')
    ax_bcharge.axhline(0, c="gray")

    ax_bcharge.set_ylabel('Power in MW')

    # SOC
    ax_soc.plot(Batt, label="Battery State of Charge")
    ax_soc.set_ylabel('Capacity in MWh')

    # Hydrogen
    ax_h2.bar(np.arange(len(Electrolyzer)),
              Electrolyzer,
              label=r"H\textsubscript{2} Charging")
    ax_h2.bar(np.arange(len(FuelCell)),
              FuelCell,
              label=r"H\textsubscript{2} Discharging")
    ax_h2.axhline(0, c="gray")

    ax_h2.set_ylabel('Power in MW')

    #LOH
    ax_loh.plot(LoH, label=r"H\textsubscript{2} Storage")
    ax_loh.set_ylabel(r'Storage in kg')

    # Common Plot Paramaters
    subplotlabels = "ABCDEFG"
    trans = mtransforms.ScaledTranslation(5 / 72, -5 / 72, fig.dpi_scale_trans)
    for i, a in enumerate(ax):
        a.legend(loc="upper right", frameon=True, edgecolor="None")
        a.text(0.0,
               1.0,
               subplotlabels[i],
               transform=a.transAxes + trans,
               va='top',
               bbox=dict(facecolor='white',
                         alpha=0.5,
                         edgecolor='none',
                         pad=3.0))

    ax[-1].set_xlim(0)
    ax[-1].set_xlabel("Hours")

    fig.align_labels()
    fig.tight_layout(h_pad=0)
    fig.savefig("output/results.pdf", transparent=True, bbox_inches="tight")
    fig.show()
