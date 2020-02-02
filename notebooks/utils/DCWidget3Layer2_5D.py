from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

from scipy.constants import epsilon_0
import copy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path

import discretize
from SimPEG import maps, utils

from pymatsolver import Pardiso
from SimPEG.utils import ExtractCoreMesh
from SimPEG.electromagnetics.static import resistivity as DC
from SimPEG.utils import ModelBuilder


from ipywidgets import FloatSlider, FloatText, ToggleButtons

from .DCLayers import widgetify

# Mesh, sigmaMap can be globals global
npad = 8
growrate = 2.0
cs = 1
hx = [(cs, npad, -growrate), (cs, 100), (cs, npad, growrate)]
hy = [(cs, npad, -growrate), (cs, 50)]
mesh = discretize.TensorMesh([hx, hy], "CN")
mapping = maps.IdentityMap(mesh)
dx = 5
xr = np.arange(-40, 41, dx)
dxr = np.diff(xr)
xmin = -40.0
xmax = 40.0
ymin = -40.0
ymax = 8.0
xylim = np.c_[[xmin, ymin], [xmax, ymax]]
indCC, meshcore = ExtractCoreMesh(xylim, mesh)
indx = (
    (mesh.gridFx[:, 0] >= xmin)
    & (mesh.gridFx[:, 0] <= xmax)
    & (mesh.gridFx[:, 1] >= ymin)
    & (mesh.gridFx[:, 1] <= ymax)
)
indy = (
    (mesh.gridFy[:, 0] >= xmin)
    & (mesh.gridFy[:, 0] <= xmax)
    & (mesh.gridFy[:, 1] >= ymin)
    & (mesh.gridFy[:, 1] <= ymax)
)
indF = np.concatenate((indx, indy))


def model_soundings(h0, h1, rho0, rho1, rho2):
    hz = np.r_[h0, h1]
    rho = np.r_[rho0, rho1, rho2]

    srcList_w = []
    srcList_s = []
    AB2 = np.arange(4, 89, 3)+0.5
    for i, a in enumerate(AB2):
        a_loc = -a
        b_loc = a
        m_loc_wen = -a + (a*2)//3
        n_loc_wen = -m_loc_wen

        m_loc_sch = -1.5
        n_loc_sch = 1.5
        rx_w = DC.Rx.Dipole(np.r_[m_loc_wen, 0, 0], np.r_[n_loc_wen, 0, 0])
        rx_s = DC.Rx.Dipole(np.r_[m_loc_sch, 0, 0], np.r_[n_loc_sch, 0, 0])

        locA = np.r_[a_loc, 0, 0]
        locB = np.r_[b_loc, 0, 0]
        src = DC.Src.Dipole([rx_w], locA, locB)
        srcList_w.append(src)
        src = DC.Src.Dipole([rx_s], locA, locB)
        srcList_s.append(src)

    m = np.r_[rho, hz]

    wires = maps.Wires(('rho', rho.size), ('t', rho.size-1))
    mapping_rho = maps.IdentityMap(nP=rho.size) * wires.rho
    mapping_t = maps.IdentityMap(nP=hz.size) * wires.t

    survey = DC.Survey(srcList_w)
    simulation = DC.DCSimulation_1D(
        rhoMap=mapping_rho,
        thicknessesMap=mapping_t,
        survey=survey,
        data_type='apparent_resistivity'
    )
    data_w = simulation.makeSyntheticData(m)

    survey = DC.Survey(srcList_s)
    simulation = DC.DCSimulation_1D(
        rhoMap=mapping_rho,
        thicknessesMap=mapping_t,
        survey=survey,
        data_type='apparent_resistivity'
    )
    data_s = simulation.makeSyntheticData(m)
    return data_w, data_s


def model_fields(A, B, h0, h1, rho0, rho1, rho2):
    # Create halfspace model
    # halfspaceMod = sig2 * np.ones([mesh.nC])
    # mhalf = np.log(halfspaceMod)
    # Create layered model with background resistivities

    resistivity_model = ModelBuilder.layeredModel(
        mesh.gridCC, np.r_[0., -h0, -(h0+h1)], np.r_[rho0, rho1, rho2]
    )

    Mx = mesh.gridCC
    # Nx = np.empty(shape=(mesh.nC, 2))
    rx = DC.Rx.Pole_ky(Mx)
    # rx = DC.Rx.Dipole(Mx,Nx)
    if B == []:
        src = DC.Src.Pole([rx], np.r_[A, 0.0])
    else:
        src = DC.Src.Dipole([rx], np.r_[A, 0.0], np.r_[B, 0.0])
    survey = DC.Survey_ky([src])
    problem = DC.Problem2D_CC(
                mesh,
                rhoMap=mapping,
                Solver=Pardiso,
                survey=survey
                )

    mesh.setCellGradBC("neumann")
    cellGrad = mesh.cellGrad
    faceDiv = mesh.faceDiv

    phi_total = problem.dpred(resistivity_model)
    e_total = -cellGrad * phi_total
    j_total = problem.MfRhoI * problem.Grad * phi_total
    q_total = epsilon_0 * problem.Vol * (faceDiv * e_total)
    total_field = {"phi": phi_total, "e": e_total, "j": j_total, "q": q_total}

    return src, total_field


def calculateRhoA(survey, VM, VN, A, B, M, N):

    eps = 1e-9  # to stabilize division

    if survey == "Dipole-Dipole":
        G = 1.0 / (
            1.0 / (np.abs(A - M) + eps)
            - 1.0 / (np.abs(M - B) + eps)
            - 1.0 / (np.abs(N - A) + eps)
            + 1.0 / (np.abs(N - B) + eps)
        )
        rho_a = (VM - VN) * 2.0 * np.pi * G
    elif survey == "Pole-Dipole":
        G = 1.0 / (1.0 / (np.abs(A - M) + eps) - 1.0 / (np.abs(N - A) + eps))
        rho_a = (VM - VN) * 2.0 * np.pi * G
    elif survey == "Dipole-Pole":
        G = 1.0 / (1.0 / (np.abs(A - M) + eps) - 1.0 / (np.abs(M - B) + eps))
        rho_a = (VM) * 2.0 * np.pi * G
    elif survey == "Pole-Pole":
        G = 1.0 / (1.0 / (np.abs(A - M) + eps))
        rho_a = (VM) * 2.0 * np.pi * G

    return rho_a


def PLOT(
    survey,
    AB2,
    h0,
    h1,
    rho0,
    rho1,
    rho2,
    Field,
    Type,
    Scale,
):
    labelsize = 16.0
    ticksize = 16.0

    survey_type = survey
    survey = 'Dipole-Dipole'

    A = -AB2
    B = AB2
    if survey_type == 'Wenner':
        M = -AB2 + (AB2*2)//3
        N = -M
    else:
        M = -1.5
        N = 1.5

    # Calculate resistivity profile
    ab2s = np.arange(4, 89, 3)+0.5
    data_w, data_s = model_soundings(h0, h1, rho0, rho1, rho2)

    mtrue = ModelBuilder.layeredModel(
        mesh.gridCC, np.r_[0., -h0, -(h0+h1)], np.r_[rho0, rho1, rho2]
    )
    if Field != 'Model':
        if Type != 'Primary':
            src, total_field = model_fields(
                A, B, h0, h1, rho0, rho1, rho2
            )
        if Type != 'Total':
            _, primary_field = model_fields(
                A, B, h0, h1, rho0, rho0, rho0
            )


    fig, ax = plt.subplots(2, 1, figsize=(11, 10))
    fig.subplots_adjust(right=0.8, wspace=0.05, hspace=0.35)

    i_a = np.argmin(np.abs(ab2s-AB2))
    if survey_type == 'Wenner':
        rhoa = data_w.dobs[i_a]
    else:
        rhoa = data_s.dobs[i_a]

    ax[0].plot(ab2s, data_w.dobs, 'r', lw=3, label='Wenner')
    ax[0].plot(ab2s, data_s.dobs, 'b', lw=3, label='Schlumberger')
    ax[0].plot([AB2], [rhoa], 'k', marker='*', markersize=labelsize)
    minrho = min(rho0, rho1, rho2)
    maxrho = max(rho0, rho1, rho2)
    if minrho>0.975*maxrho:
        ax[0].set_ylim([0.975*rho1, 1.025*rho1])

    xytext = (AB2+1, rhoa)
    ax[0].annotate(r"$\rho_a$ = {:.2f}".format(rhoa), xy=xytext, fontsize=labelsize)

    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend()
    ax[0].set_ylabel(r'$\rho_a$ ($\Omega$m)')
    ax[0].set_xlabel(r'$\frac{AB}{2}$ (m)')
    matplotlib.rcParams['font.size'] = labelsize
    ax[0].grid(True, which="both", ls="--", c='gray')

    if Field == "Model":

        label = "Resisitivity (ohm-m)"
        xtype = "CC"
        view = "real"
        streamOpts = None
        ind = indCC

        formatter = "%.1e"
        pcolorOpts = {"cmap": "jet_r"}
        if Scale == "Log":
            pcolorOpts = {"norm": matplotlib.colors.LogNorm(), "cmap": "jet_r"}

        u = mtrue

    elif Field == "Potential":

        label = "Potential (V)"
        xtype = "CC"
        view = "real"
        streamOpts = None
        ind = indCC

        formatter = "%.1e"
        pcolorOpts = {"cmap": "viridis"}
        if Scale == "Log":
            linthresh = 10.0
            pcolorOpts = {
                "norm": matplotlib.colors.SymLogNorm(linthresh=linthresh, linscale=0.2),
                "cmap": "viridis",
            }
        if Type == 'Total':
            u = total_field["phi"]
        if Type == 'Primary':
            u = primary_field['phi']
        elif Type == 'Secondary':
            u = total_field["phi"] - primary_field['phi']

    elif Field == "E":

        label = "Electric Field (V/m)"
        xtype = "F"
        view = "vec"
        streamOpts = {"color": "w"}
        ind = indF

        # formatter = LogFormatter(10, labelOnlyBase=False)
        pcolorOpts = {"cmap": "viridis"}
        if Scale == "Log":
            pcolorOpts = {"norm": matplotlib.colors.LogNorm(), "cmap": "viridis"}
        formatter = "%.1e"

        if Type == "Total":
            u = total_field["e"]

        elif Type == "Primary":
            u = primary_field["e"]

        elif Type == "Secondary":
            uTotal = total_field["e"]
            uPrim = primary_field["e"]
            u = uTotal - uPrim

    elif Field == "J":

        label = "Current density ($A/m^2$)"
        xtype = "F"
        view = "vec"
        streamOpts = {"color": "w"}
        ind = indF

        # formatter = LogFormatter(10, labelOnlyBase=False)
        pcolorOpts = {"cmap": "viridis"}
        if Scale == "Log":
            pcolorOpts = {"norm": matplotlib.colors.LogNorm(), "cmap": "viridis"}
        formatter = "%.1e"

        if Type == "Total":
            u = total_field["j"]

        elif Type == "Primary":
            u = primary_field["j"]

        elif Type == "Secondary":
            uTotal = total_field["j"]
            uPrim = primary_field["j"]
            u = uTotal - uPrim

    elif Field == "Charge":

        label = "Charge Density ($C/m^2$)"
        xtype = "CC"
        view = "real"
        streamOpts = None
        ind = indCC

        # formatter = LogFormatter(10, labelOnlyBase=False)
        pcolorOpts = {"cmap": "RdBu_r"}
        if Scale == "Log":
            linthresh = 1e-12
            pcolorOpts = {
                "norm": matplotlib.colors.SymLogNorm(linthresh=linthresh, linscale=0.2),
                "cmap": "RdBu_r",
            }
        formatter = "%.1e"

        if Type == "Total":
            u = total_field["q"]

        elif Type == "Primary":
            u = primary_field["q"]

        elif Type == "Secondary":
            uTotal = total_field["q"]
            uPrim = primary_field["q"]
            u = uTotal - uPrim

    if Scale == "Log":
        eps = 1e-16
    else:
        eps = 0.0
    dat = meshcore.plotImage(
        u[ind] + eps,
        vType=xtype,
        ax=ax[1],
        grid=False,
        view=view,
        streamOpts=streamOpts,
        pcolorOpts=pcolorOpts,
    )

    ax[1].set_xlabel("x (m)", fontsize=labelsize)
    ax[1].set_ylabel("z (m)", fontsize=labelsize)

    ax[1].plot(A, 1.0, marker="v", color="red", markersize=labelsize)
    ax[1].plot(B, 1.0, marker="v", color="blue", markersize=labelsize)
    ax[1].plot(M, 1.0, marker="^", color="yellow", markersize=labelsize)
    ax[1].plot(N, 1.0, marker="^", color="green", markersize=labelsize)

    xytextA1 = (A - 0.5, 2.5)
    xytextB1 = (B - 0.5, 2.5)
    xytextM1 = (M - 0.5, 2.5)
    xytextN1 = (N - 0.5, 2.5)
    ax[1].annotate("A", xy=xytextA1, xytext=xytextA1, fontsize=labelsize)
    ax[1].annotate("B", xy=xytextB1, xytext=xytextB1, fontsize=labelsize)
    ax[1].annotate("M", xy=xytextM1, xytext=xytextM1, fontsize=labelsize)
    ax[1].annotate("N", xy=xytextN1, xytext=xytextN1, fontsize=labelsize)

    ax[1].tick_params(axis="both", which="major", labelsize=ticksize)
    cbar_ax = fig.add_axes([0.65, 0.05, 0.08, 0.5])
    cbar_ax.axis("off")
    vmin, vmax = dat[0].get_clim()
    if Scale == "Log":

        if (Field == "E") or (Field == "J"):
            cb = plt.colorbar(
                dat[0],
                ax=cbar_ax,
                format=formatter,
                ticks=np.logspace(np.log10(vmin), np.log10(vmax), 5),
            )

        elif Field == "Model":
            cb = plt.colorbar(
                dat[0],
                ax=cbar_ax,
                format=formatter,
                ticks=np.logspace(np.log10(vmin), np.log10(vmax), 5),
            )

        else:
            cb = plt.colorbar(
                dat[0],
                ax=cbar_ax,
                format=formatter,
                ticks=np.r_[
                    -1.0
                    * np.logspace(np.log10(-vmin - eps), np.log10(linthresh), 3)[:-1],
                    0.0,
                    np.logspace(np.log10(linthresh), np.log10(vmax), 3)[1:],
                ],
            )
    else:
        cb = plt.colorbar(
            dat[0], ax=cbar_ax, format=formatter, ticks=np.linspace(vmin, vmax, 5)
        )
    # t_logloc = matplotlib.ticker.LogLocator(base=10.0, subs=[1.0,2.], numdecs=4, numticks=8)
    # tick_locator = matplotlib.ticker.SymmetricalLogLocator(t_logloc)
    # cb.locator = tick_locator
    # cb.ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
    # cb.update_ticks()
    cb.ax.tick_params(labelsize=ticksize)
    cb.set_label(label, fontsize=labelsize)
    cb.minorticks_off()
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    ax[1].set_aspect("equal")

    plt.show()


def ThreeLayer_app():
    app = widgetify(
        PLOT,
        survey=ToggleButtons(
            options=["Wenner", "Schlumberger"],
            value="Wenner",
        ),
        AB2=FloatSlider(
            min=4.5,
            max=88.5,
            step=3,
            value=19.5,
            continuous_update=False,
            description="$\\frac{AB}{2}$"
        ),
        h0=FloatSlider(
            min=0.0,
            max=20.0,
            step=1.0,
            value=10.0,
            continuous_update=False,
            description="$h_1$",
        ),
        h1=FloatSlider(
            min=0,
            max=20.0,
            step=1.0,
            value=10.0,
            continuous_update=False,
            description="$h_2$",
        ),
        rho0=FloatText(
            min=1e-8,
            max=1e8,
            value=5000.0,
            continuous_update=False,
            description="$\\rho_{1}$",
        ),
        rho1=FloatText(
            min=1e-8,
            max=1e8,
            value=500.0,
            continuous_update=False,
            description="$\\rho_{2}$",
        ),
        rho2=FloatText(
            min=1e-8,
            max=1e8,
            value=5000.0,
            continuous_update=False,
            description="$\\rho_{3}$",
        ),
        Field=ToggleButtons(
            options=["Model", "Potential", "E", "J", "Charge"],
            value="Model",
        ),
        Type=ToggleButtons(options=["Total", "Primary", "Secondary"], value="Total"),
        Scale=ToggleButtons(options=["Linear", "Log"], value="Log"),
    )
    return app
