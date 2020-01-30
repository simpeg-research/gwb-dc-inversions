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


from ipywidgets import FloatSlider, FloatText, ToggleButtons

from .DCLayers import widgetify

# Mesh, sigmaMap can be globals global
npad = 20
growrate = 2.0
cs = 0.5
hx = [(cs, npad, -growrate), (cs, 200), (cs, npad, growrate)]
hy = [(cs, npad, -growrate), (cs, 100)]
mesh = discretize.TensorMesh([hx, hy], "CN")
mapping = maps.IdentityMap(mesh)
dx = 5
xr = np.arange(-40, 41, dx)
dxr = np.diff(xr)
xmin = -30.0
xmax = 30.0
ymin = -30.0
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

rhoa_min = None
rhoa_max = None
last_rhohalf = None
last_rhoTarget = None
last_survey_type = None


def model_fields(survey_type, a_spacing, array_center, xc, zc, r, rhoHalf, rhoTarget):
    # Create halfspace model
    mhalf = rhoHalf * np.ones(mesh.nC)

    grid_r = np.sqrt((xc-mesh.gridCC[:, 0])**2 + (zc-mesh.gridCC[:, 1])**2)

    # Add plate or cylinder
    mtrue = mhalf.copy()
    mtrue[grid_r < r] = rhoTarget

    Mx = mesh.gridCC
    # Nx = np.empty(shape=(mesh.nC, 2))
    rx = DC.Rx.Pole_ky(Mx)
    # rx = DC.Rx.Dipole(Mx,Nx)

    if survey_type == 'Wenner':
        A = array_center - 1.5*a_spacing
        M = array_center - 0.5*a_spacing
        N = array_center + 0.5*a_spacing
        B = array_center + 1.5*a_spacing
    elif survey_type == 'Dipole-Dipole':
        A = array_center - 1.5*a_spacing
        B = array_center - 0.5*a_spacing
        M = array_center + 0.5*a_spacing
        N = array_center + 1.5*a_spacing

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

    phi_total = problem.dpred(mtrue)
    e_total = -cellGrad * phi_total
    j_total = problem.MfRhoI * problem.Grad * phi_total
    q_total = epsilon_0 * problem.Vol * (faceDiv * e_total)
    total_field = {"phi": phi_total, "e": e_total, "j": j_total, "q": q_total}

    return src, total_field, [A, B, M, N]


def get_Surface_Potentials(survey, src, field_obj):

    phi = field_obj["phi"]
    CCLoc = mesh.gridCC
    zsurfaceLoc = np.max(CCLoc[:, 1])
    surfaceInd = np.where(CCLoc[:, 1] == zsurfaceLoc)
    xSurface = CCLoc[surfaceInd, 0].T
    phiSurface = phi[surfaceInd]
    phiScale = 0.0

    if survey == "Pole-Dipole" or survey == "Pole-Pole":
        refInd = utils.closestPoints(mesh, [xmax + 60.0, 0.0], gridLoc="CC")
        phiScale = phi[refInd]
        phiSurface = phiSurface - phiScale

    return xSurface, phiSurface, phiScale


def getSensitivity(survey, A, B, M, N, model):

    if survey == "Dipole-Dipole":
        rx = DC.Rx.Dipole_ky(np.r_[M, 0.0], np.r_[N, 0.0])
        src = DC.Src.Dipole([rx], np.r_[A, 0.0], np.r_[B, 0.0])
    elif survey == "Pole-Dipole":
        rx = DC.Rx.Dipole_ky(np.r_[M, 0.0], np.r_[N, 0.0])
        src = DC.Src.Pole([rx], np.r_[A, 0.0])
    elif survey == "Dipole-Pole":
        rx = DC.Rx.Pole_ky(np.r_[M, 0.0])
        src = DC.Src.Dipole([rx], np.r_[A, 0.0], np.r_[B, 0.0])
    elif survey == "Pole-Pole":
        rx = DC.Rx.Pole_ky(np.r_[M, 0.0])
        src = DC.Src.Pole([rx], np.r_[A, 0.0])

    survey = DC.Survey_ky([src])
    problem = DC.Problem2D_CC(
                mesh,
                sigmaMap=mapping,
                Solver=Pardiso,
                survey=survey
                )
    fieldObj = problem.fields(model)

    J = problem.Jtvec(model, np.array([1.0]), f=fieldObj)

    return J

def q_on_cyl(I_loc, rad, n_sides, ys, center, sig_back, sig_cyl):

    d_side = 2*np.sin(np.pi/n_sides)*rad
    temp = np.linspace(-np.pi, np.pi, n_sides+1)
    theta_grid = (temp[1:]+temp[:-1])/2

    hs = ys[1:]-ys[:-1]
    grid_y = (ys[1:]+ys[:-1])/2

    areas = np.outer(hs, d_side*np.ones(n_sides)) #(n_sides, n_y)
    normals = np.c_[np.cos(theta_grid), np.sin(theta_grid)] #y normal is 0
    grid_x = rad*normals[:,0] + center[0]
    grid_z = rad*normals[:,1] + center[1]

    X, Y = np.meshgrid(grid_x, grid_y)
    Z, _ = np.meshgrid(grid_z, grid_y)

    # each point on the surface of the cylinder should now have a grid_x, grid_y, grid_z center location

    # Ex and Ez at each point on the surface of the cylinder
    Xs = I_loc[0]-X
    Ys = I_loc[1]-Y
    Zs = I_loc[2]-Z
    R = np.sqrt(Xs**2 + Ys**2 + Zs**2)
    R3 = R*R*R

    Ex = 1.0/(sig_back*2*np.pi)*(Xs/R3)  # x_gradient of phi
    Ez = 1.0/(sig_back*2*np.pi)*(Zs/R3)  # z_gradient of phi

    dot = Ex*normals[:, 0] + Ez*normals[:, 1]
    scale = (sig_back - sig_cyl)/sig_cyl
    tau_eps = scale*dot
    q_eps = areas*tau_eps
    return 2*q_eps, (X, Y, Z) # Double it because it's symmetric!


def secondary_potential(q, q_grid, electrode_locs):
    X, Y, Z = q_grid
    q = q.reshape(-1)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)

    x_obs, y_obs, z_obs = electrode_locs
    #grid of Xs, Ys, Zs for everything
    Xs = X[:, None] - x_obs
    Ys = Y[:, None] - y_obs
    Zs_p = Z[:, None] - z_obs
    Zs_i = -Z[:, None] - z_obs

    R_p = np.sqrt(Xs**2 + Ys**2 + Zs_p**2)
    R_i = np.sqrt(Xs**2 + Ys**2 + Zs_i**2)

    pot_p = np.sum(q[:,None]/R_p, axis=0)
    pot_i = np.sum(q[:,None]/R_i, axis=0)
    return (pot_p+pot_i)/(4*np.pi)

def primary_potential(A, electrode_locations, sig_back):
    x, y, z = electrode_locations
    X = A[0]-x
    Y = A[1]-y
    Z = A[2]-z
    R = np.sqrt(X**2 + Y**2 + Z**2)

    return 1/(2*np.pi*sig_back*R)

def calculateRhoA(survey, VM, VN, A, B, M, N):

    eps = 1e-9  # to stabilize division

    source, receive = survey.split('-')
    bot = 1.0 / (np.abs(A - M) + eps)
    if source == 'Dipole':
        bot -= 1.0 / (np.abs(M - B) + eps)
    if receive == 'Dipole':
        bot -= 1.0 / (np.abs(N - A) + eps)
        if source == 'Dipole':
            bot += 1.0 / (np.abs(N - B) + eps)
    G = 1.0 / bot

    if receive == 'Dipole':
        return (VM - VN)*2*np.pi*G
    return (VM)*2*np.pi*G


def PLOT(
    survey_type,
    a_spacing,
    array_center,
    xc,
    zc,
    r,
    rhohalf,
    rhoTarget,
    Field,
    Scale,
):
    global rhoa_min, rhoa_max, last_rhohalf, last_rhoTarget, last_survey_type
    labelsize = 16.0
    ticksize = 16.0

    survey = 'Dipole-Dipole'

    if survey_type == 'Wenner':
        A = array_center - 1.5*a_spacing
        M = array_center - 0.5*a_spacing
        N = array_center + 0.5*a_spacing
        B = array_center + 1.5*a_spacing
    elif survey_type == 'Dipole-Dipole':
        A = array_center - 1.5*a_spacing
        B = array_center - 0.5*a_spacing
        M = array_center + 0.5*a_spacing
        N = array_center + 1.5*a_spacing

    if np.abs(zc)-2<r:
        r = np.abs(zc)-2
        print(f'set r = {r}')

    mtrue = np.ones(mesh.nC)*rhohalf
    grid_r = np.sqrt((xc-mesh.gridCC[:, 0])**2 + (zc-mesh.gridCC[:, 1])**2)
    mtrue[grid_r < r] = rhoTarget

    sig_back = 1.0/rhohalf
    sig_cyl = 1.0/rhoTarget
    n_sides = 25
    n_y = 100
    ys = np.r_[0, np.cumsum(np.logspace(-3, 2, n_y))]

    # Calculate profile from first order charge accumulation
    x_centers = np.linspace(-30.25, 30.25, 122)
    profile_rhoa = np.empty_like(x_centers)
    for ii, x in enumerate(x_centers):
        if survey_type == 'Wenner':
            A_i = x - 1.5*a_spacing
            M_i = x - 0.5*a_spacing
            N_i = x + 0.5*a_spacing
            B_i = x + 1.5*a_spacing
        elif survey_type == 'Dipole-Dipole':
            A_i = x - 1.5*a_spacing
            B_i = x - 0.5*a_spacing
            M_i = x + 0.5*a_spacing
            N_i = x + 1.5*a_spacing

        qA, q_grid = q_on_cyl([A_i,0,0], r, n_sides, ys, [xc, zc], sig_back, sig_cyl)
        qB, q_grid = q_on_cyl([B_i,0,0], r, n_sides, ys, [xc, zc], sig_back, sig_cyl)
        q = qA-qB

        loc_x = [M_i, N_i]
        loc_y = [0, 0]
        loc_z = [0, 0]
        v_s = secondary_potential(q, q_grid, (loc_x, loc_y, loc_z))
        v_a = primary_potential(np.r_[A_i,0,0], (loc_x, loc_y, loc_z), sig_back)
        v_b = primary_potential(np.r_[B_i,0,0], (loc_x, loc_y, loc_z), sig_back)
        v_p = v_a-v_b

        V = v_p+v_s
        VM = V[0]
        VN = V[1]

        profile_rhoa[ii] = calculateRhoA(survey, VM, VN, A_i, B_i, M_i, N_i)

    if Field != 'Model':
        src, total_field, array = model_fields(
            survey_type, a_spacing, array_center, xc, zc, r, rhohalf, rhoTarget
        )

        xSurface, phiTotalSurface, phiScaleTotal = get_Surface_Potentials(
            survey, src, total_field
        )
        MInd = np.where(xSurface == M)
        NInd = np.where(xSurface == N)

        VM = phiTotalSurface[MInd[0]]
        VN = phiTotalSurface[NInd[0]]

    fig, ax = plt.subplots(2, 1, figsize=(9 * 1.5, 9 * 1.8), sharex=True)
    fig.subplots_adjust(right=0.8, wspace=0.05, hspace=0.05)

    prof_min, prof_max = profile_rhoa.min(), 1.02*profile_rhoa.max()
    prof_min -= 0.01*prof_min
    prof_max += 0.01*prof_max
    if (
            rhohalf != last_rhohalf or
            rhoTarget != last_rhoTarget or
            last_survey_type != survey_type
    ):
        last_rhohalf = rhohalf
        last_rhoTarget = rhoTarget
        last_survey_type = survey_type
        rhoa_min, rhoa_max = prof_min, prof_max

    if rhoa_min > prof_min:
        rhoa_min = prof_min
    if rhoa_max < prof_max:
        rhoa_max = prof_max

    rhoa = profile_rhoa[x_centers==array_center]

    ax[0].plot(x_centers, profile_rhoa)
    ax[0].plot([array_center], [rhoa], marker='*', markersize=labelsize)
    xytext = (array_center, rhoa*1.01)
    ax[0].annotate(r"$\rho_a$ = {:.2f}".format(rhoa[0]), xy=xytext, fontsize=labelsize)
    #ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([rhoa_min, rhoa_max])
    ax[0].set_ylabel(r'$\rho_a$')
    #ax[0].set_yscale('log')

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

        u = (mapping * mtrue)

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
        u = total_field["phi"] - phiScaleTotal

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

        u = total_field["e"]

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

        u = total_field["j"]

    elif Field == "Sensitivity":

        label = "Sensitivity"
        xtype = "CC"
        view = "real"
        streamOpts = None
        ind = indCC

        # formatter = None
        # pcolorOpts = {"cmap":"viridis"}
        # formatter = LogFormatter(10, labelOnlyBase=False)
        pcolorOpts = {"cmap": "viridis"}
        if Scale == "Log":
            linthresh = 1e-4
            pcolorOpts = {
                "norm": matplotlib.colors.SymLogNorm(linthresh=linthresh, linscale=0.2),
                "cmap": "viridis",
            }
        # formatter = formatter = "$10^{%.1f}$"
        formatter = "%.1e"

        u = getSensitivity(survey, A, B, M, N, mtrue)

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
    if rhoTarget != rhohalf:
        from matplotlib.patches import Circle
        circle = Circle((xc, zc), r, fill=False, color='k', linestyle='--')
        ax[1].add_patch(circle)

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
    cbar_ax = fig.add_axes([0.8, 0.05, 0.08, 0.5])
    cbar_ax.axis("off")
    vmin, vmax = dat[0].get_clim()
    if Scale == "Log":
        if (Field == "E") or (Field == "J") or (Field == 'Model'):
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

    cb.ax.tick_params(labelsize=ticksize)
    cb.minorticks_off()
    cb.set_label(label, fontsize=labelsize)
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    ax[1].set_aspect("equal")

    plt.show()


def ResCylLayer_app():
    app = widgetify(
        PLOT,
        survey_type=ToggleButtons(
            options=["Wenner", "Dipole-Dipole"],
            value="Wenner",
        ),
        a_spacing=FloatSlider(
            min=2,
            max=10.0,
            step=1.0,
            value=4.0,
            continuous_update=False,
            description="a",
        ),
        array_center=FloatSlider(
            min=-30.25,
            max=30.25,
            step=0.5,
            value=-10.25,
            continuous_update=False,
            description="array center",
        ),
        xc=FloatSlider(
            min=-30.0, max=30.0, step=1.0, value=0.0, continuous_update=False
        ),
        zc=FloatSlider(
            min=-30.0, max=-3.0, step=0.5, value=-10, continuous_update=False
        ),
        r=FloatSlider(min=1.0, max=10.0, step=0.5, value=8.0, continuous_update=False),
        rhohalf=FloatText(
            min=1e-8,
            max=1e8,
            value=500.0,
            continuous_update=False,
            description="$\\rho_{half}$",
        ),
        rhoTarget=FloatText(
            min=1e-8,
            max=1e8,
            value=50.0,
            continuous_update=False,
            description="$\\rho_{cyl}$",
        ),
        Field=ToggleButtons(
            options=["Model", "Potential", "E", "J"],
            value="Model",
        ),
        Scale=ToggleButtons(options=["Linear", "Log"], value="Log"),
    )
    return app
