from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np

import ipywidgets
from ipywidgets import FloatSlider, FloatText, ToggleButtons

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib import rcParams

import discretize
from SimPEG import utils
from pymatsolver import Pardiso

rcParams["font.size"] = 16

# Mesh parameters
npad = 20
cs = 0.5
hx = [(cs, npad, -1.3), (cs, 200), (cs, npad, 1.3)]
hy = [(cs, npad, -1.3), (cs, 100)]
mesh = discretize.TensorMesh([hx, hy], "CN")

# bounds on electrical resistivity
rhomin = 1e2
rhomax = 1e3

eps = 1e-9  # to stabilize division
infinity = 100  # what is "far enough"


def r(xyz, src_loc):
    """
    Distance from source to points on an xyz grid
    """
    return (
        np.sqrt(
            (xyz[:, 0] - src_loc[0]) ** 2
            + (xyz[:, 1] - src_loc[1]) ** 2
            + (xyz[:, 2] - src_loc[2]) ** 2
        )
        + eps
    )


def sum_term(rho1, rho2, h, r):
    m = utils.mkvc(np.arange(1, infinity + 1))
    k = (rho2 - rho1) / (rho2 + rho1)
    return np.sum(
        ((k ** m.T) * np.ones_like(utils.mkvc(r, 2)))
        / np.sqrt(1.0 + (2.0 * h * m.T / utils.mkvc(r, 2)) ** 2),
        1,
    )


def sum_term_deriv(rho1, rho2, h, r):
    m = utils.mkvc(np.arange(1, infinity + 1))
    k = (rho2 - rho1) / (rho2 + rho1)
    return np.sum(
        ((k ** m.T) * np.ones_like(utils.mkvc(r, 2)))
        / (1.0 + (2.0 * h * m.T / utils.mkvc(r, 2)) ** 2) ** (3.0 / 2.0)
        * ((2.0 * h * m.T) ** 2 / utils.mkvc(r, 2) ** 3),
        1,
    )


def layer_potentials(rho1, rho2, h, A, B, xyz):

    """
    Compute analytic solution of surface potential for 2-layered Earth
    (Ref Telford 1990, section 8.3.4)s
    """

    def V(I, src_loc):
        return (I * rho1 / (2.0 * np.pi * r(xyz, src_loc))) * (
            1 + 2 * sum_term(rho1, rho2, h, r(xyz, src_loc))
        )

    pot = V(1.0, A)
    if B is not None:
        pot += V(-1.0, B)

    return pot


def layer_E(rho1, rho2, h, A, B, xyz):
    def dr_dx(src_loc):
        return (xyz[:, 0] - src_loc[0]) / r(xyz, src_loc)

    def dr_dy(src_loc):
        return (xyz[:, 1] - src_loc[1]) / r(xyz, src_loc)

    def dr_dz(src_loc):
        return (xyz[:, 2] - src_loc[2]) / r(xyz, src_loc)

    # m = utils.mkvc(np.arange(1, infinity + 1))

    def deriv_1(r):
        return (-1.0 / r) * (1.0 + 2.0 * sum_term(rho1, rho2, h, r))

    def deriv_2(r):
        return 2.0 * sum_term_deriv(rho1, rho2, h, r)

    def Er(I, r):
        return -(I * rho1 / (2.0 * np.pi * r)) * (deriv_1(r) + deriv_2(r))

    def Ex(I, src_loc):
        return Er(I, r(xyz, src_loc)) * dr_dx(src_loc)

    def Ey(I, src_loc):
        return Er(I, r(xyz, src_loc)) * dr_dy(src_loc)

    def Ez(I, src_loc):
        return Er(I, r(xyz, src_loc)) * dr_dz(src_loc)

    ex = Ex(1.0, A)
    ey = Ey(1.0, A)
    ez = Ez(1.0, A)
    if B is not None:
        ex += Ex(-1.0, B)
        ey += Ey(-1.0, B)
        ez += Ez(-1.0, B)

    return ex, ey, ez


def layer_J(rho1, rho2, h, A, B, xyz):
    ex, ey, ez = layer_E(rho1, rho2, h, A, B, xyz)

    sig = 1.0 / rho2 * np.ones_like(xyz[:, 0])

    # print sig
    sig[xyz[:, 1] >= -h] = 1.0 / rho1  # since the model is 2D

    return sig * ex, sig * ey, sig * ez


def G(A, B, M, N):
    """
    Geometric factor
    """
    bot = 1.0 / (np.abs(A[0] - M) + eps)
    if B is not None:
        bot -= 1.0 / (np.abs(M - B[0]) + eps)
    if N is not None:
        bot -= 1.0 / (np.abs(N - A[0]) + eps)
        if B is not None:
            bot += 1.0 / (np.abs(N - B[0]) + eps)
    return 1.0 / bot


def rho_a(VM, VN, A, B, M, N):
    """
    Apparent Resistivity
    """
    if VN is None:
        return VM * 2.0 * np.pi * G(A, B, M, None)
    else:
        return (VM - VN) * 2.0 * np.pi * G(A, B, M, N)


def solve_2D_potentials(rho1, rho2, h, A, B):
    """
    Here we solve the 2D DC problem for potentials (using SimPEG Mesh Class)
    """
    sigma = 1.0 / rho2 * np.ones(mesh.nC)
    sigma[mesh.gridCC[:, 1] >= -h] = 1.0 / rho1  # since the model is 2D

    q = np.zeros(mesh.nC)
    a = utils.closestPoints(mesh, A[:2])
    q[a] = 1.0 / mesh.vol[a]
    if B is not None:
        b = utils.closestPoints(mesh, B[:2])
        q[b] = -1.0 / mesh.vol[b]


    # Use a Neumann Boundary Condition (so pole source is reasonable)
    fxm, fxp, fym, fyp = mesh.faceBoundaryInd

    n_xm = fxm.sum()
    n_xp = fxp.sum()
    n_ym = fym.sum()
    n_yp = fyp.sum()

    xBC_xm = np.zeros(n_xm) # 0.5*a_xm
    xBC_xp = np.zeros(n_xp) # 0.5*a_xp/b_xp
    yBC_xm = np.ones(n_xm) # 0.5*(1.-b_xm)
    yBC_xp = np.ones(n_xp) # 0.5*(1.-1./b_xp)
    xBC_ym = np.zeros(n_ym) # 0.5*a_ym
    xBC_yp = np.zeros(n_yp) # 0.5*a_yp/b_yp
    yBC_ym = np.ones(n_ym) # 0.5*(1.-b_ym)
    yBC_yp = np.ones(n_yp) # 0.5*(1.-1./b_yp)

    sortindsfx = np.argsort(np.r_[np.arange(mesh.nFx)[fxm],
                                  np.arange(mesh.nFx)[fxp]])
    sortindsfy = np.argsort(np.r_[np.arange(mesh.nFy)[fym],
                                  np.arange(mesh.nFy)[fyp]])
    xBC_x = np.r_[xBC_xm, xBC_xp][sortindsfx]
    xBC_y = np.r_[xBC_ym, xBC_yp][sortindsfy]
    yBC_x = np.r_[yBC_xm, yBC_xp][sortindsfx]
    yBC_y = np.r_[yBC_ym, yBC_yp][sortindsfy]
    x_BC = np.r_[xBC_x, xBC_y]
    y_BC = np.r_[yBC_x, yBC_y]

    V = utils.sdiag(mesh.vol)
    Div = V * mesh.faceDiv
    P_BC, B = mesh.getBCProjWF_simple()
    M = B*mesh.aveCC2F
    Grad = Div.T - P_BC*utils.sdiag(y_BC)*M

    A = (
        Div
        * utils.sdiag(1.0 / (mesh.dim * mesh.aveF2CC.T * (1.0 / sigma)))
        * Grad
    )

    A[0, 0] = A[0, 0] + 1. # Because Neumann
    Ainv = Pardiso(A)

    V = Ainv * q
    return V


def solve_2D_E(rho1, rho2, h, A, B):
    """
    solve the 2D DC resistivity problem for electric fields
    """

    V = solve_2D_potentials(rho1, rho2, h, A, B)
    E = -mesh.cellGrad * V
    E = mesh.aveF2CCV * E
    ex = E[: mesh.nC]
    ez = E[mesh.nC :]
    return ex, ez, V


def solve_2D_J(rho1, rho2, h, A, B):

    ex, ez, V = solve_2D_E(rho1, rho2, h, A, B)
    sigma = 1.0 / rho2 * np.ones(mesh.nC)
    sigma[mesh.gridCC[:, 1] >= -h] = 1.0 / rho1  # since the model is 2D

    return utils.sdiag(sigma) * ex, utils.sdiag(sigma) * ez, V


def plot_layer_potentials(survey_type, rho1, rho2, h, A, B, M, N, imgplt="Model"):

    markersize = 8.0
    fontsize = 16.0
    ylim = np.r_[-1.0, 1.0] * rhomax / (5 * 2 * np.pi) * 1.5

    fig, ax = plt.subplots(2, 1, figsize=(9, 7))

    fig.subplots_adjust(right=0.8)
    x = np.linspace(-40.0, 40.0, 200)
    z = np.linspace(x.min(), 0, 100)

    pltgrid = utils.ndgrid(x, z)
    xplt = pltgrid[:, 0].reshape(x.size, z.size, order="F")
    zplt = pltgrid[:, 1].reshape(x.size, z.size, order="F")

    source_type, receiver_type = survey_type.split('-')
    dipole_source = source_type == 'Dipole'
    dipole_receiver = receiver_type == 'Dipole'

    A_loc = np.r_[A, 0.0, 0.0]
    if dipole_source:
        B_loc = np.r_[B, 0.0, 0.0]
    else:
        B_loc = None

    V = layer_potentials(
        rho1,
        rho2,
        h,
        A_loc,
        B_loc,
        utils.ndgrid(x, np.r_[0.0], np.r_[0.0]),
    )
    VM = layer_potentials(
        rho1,
        rho2,
        h,
        A_loc,
        B_loc,
        utils.mkvc(np.r_[M, 0.0, 0], 2).T,
    )
    if dipole_receiver:
        VN = layer_potentials(
            rho1,
            rho2,
            h,
            A_loc,
            B_loc,
            utils.mkvc(np.r_[N, 0.0, 0], 2).T,
        )
    else:
        VN = None

    ax[0].plot(x, V, color=[0.1, 0.5, 0.1], linewidth=2)
    ax[0].grid(
        which="both", linestyle="-", linewidth=0.5, color=[0.2, 0.2, 0.2], alpha=0.5
    )
    ax[0].plot(A, 0, "+", markersize=12, markeredgewidth=3, color=[1.0, 0.0, 0])
    if dipole_source:
        ax[0].plot(B, 0, "_", markersize=12, markeredgewidth=3, color=[0.0, 0.0, 1.0])
    ax[0].set_ylabel("Potential, (V)")
    ax[0].set_xlabel("x (m)")
    ax[0].set_xlim([x.min(), x.max()])
    ax[0].set_ylim(ylim)

    ax[0].plot(M, VM, "o", color="k")
    if dipole_receiver:
        ax[0].plot(N, VN, "o", color="k")

    props = dict(boxstyle="round", facecolor="grey", alpha=0.3)

    txtsp = 1

    xytextM = (M + 0.5, np.max([np.min([VM, ylim.max()]), ylim.min()]) + 0.5)
    if dipole_receiver:
        xytextN = (N + 0.5, np.max([np.min([VN, ylim.max()]), ylim.min()]) + 0.5)

    props = dict(boxstyle="round", facecolor="grey", alpha=0.4)

    ax[0].annotate("%2.1e" % (VM), xy=xytextM, xytext=xytextM)
    if dipole_receiver:
        ax[0].annotate("%2.1e" % (VN), xy=xytextN, xytext=xytextN)

    # ax[0].plot(np.r_[M, N], np.ones(2)*VN, color='k')
    # ax[0].plot(np.r_[M, M], np.r_[VM, VN], color='k')
    # ax[0].annotate('%2.1e'%(VM-VN) , xy=(M, (VM+VN)/2), xytext=(M-9, (VM+VN)/2.))

    props = dict(boxstyle="round", facecolor="grey", alpha=0.4)
    ax[0].text(
        x.max() + 1,
        ylim.max() - 0.1 * ylim.max(),
        "$\\rho_a$ = %2.2f" % (rho_a(VM, VN, A_loc, B_loc, M, N)),
        verticalalignment="bottom",
        bbox=props,
    )

    match_point = 0 # np.searchsorted(x, M) #  make potential match at point A
    if imgplt == "Model":
        model = rho2 * np.ones(pltgrid.shape[0])
        model[pltgrid[:, 1] >= -h] = rho1
        model = model.reshape(x.size, z.size, order="F")
        cb = ax[1].pcolor(xplt, zplt, model, norm=LogNorm())
        ax[1].plot(
            [xplt.min(), xplt.max()],
            -h * np.r_[1.0, 1],
            color=[0.5, 0.5, 0.5],
            linewidth=1.5,
        )

        clim = [rhomin, rhomax]
        clabel = "Resistivity ($\Omega$m)"

    # elif imgplt == 'potential':
    #     Vplt = layer_potentials(rho1, rho2, h, np.r_[A, 0., 0.], np.r_[B, 0., 0.], np.c_[pltgrid, np.zeros_like(pltgrid[:, 0])])
    #     Vplt = Vplt.reshape(x.size, z.size, order='F')
    #     cb = ax[1].pcolor(xplt, zplt, Vplt)
    #     ax[1].contour(xplt, zplt, np.abs(Vplt), np.logspace(-2., 1., 10), colors='k', alpha=0.5)
    #     ax[1].set_ylabel('z (m)', fontsize=16)
    #     clim = ylim
    #     clabel = 'Potential (V)'

    elif imgplt == "Potential":
        Pc = mesh.getInterpolationMat(pltgrid, "CC")

        V = solve_2D_potentials(rho1, rho2, h, A_loc, B_loc)

        Vplt = Pc * V
        Vplt = Vplt.reshape(x.size, z.size, order="F")

        # since we are using a strictly 2D code, the potnetials at the surface
        # do not match the analytic, so we scale the potentials to match the
        # analytic 2.5D result at the surface.
        fudgeFactor = layer_potentials(
            rho1,
            rho2,
            h,
            A_loc,
            B_loc,
            np.c_[x[match_point], 0.0, 0.0],
        )/Vplt[match_point, 0]

        norm = SymLogNorm(linthresh=15)#vmin=-1.0, vmax=1.0)

        cb = ax[1].pcolormesh(xplt, zplt, Vplt * fudgeFactor, cmap="viridis",
                norm=norm)
        ax[1].plot(
            [xplt.min(), xplt.max()],
            -h * np.r_[1.0, 1],
            color=[0.5, 0.5, 0.5],
            linewidth=1.5,
        )
        ax[1].contour(xplt, zplt, np.abs(Vplt), colors="k", alpha=0.5)
        ax[1].set_ylabel("z (m)", fontsize=16)
        #if dipole_source:
        #    clim = np.r_[-15.0, 15.0]
        clabel = "Potential (V)"

    elif imgplt == "E":

        Pc = mesh.getInterpolationMat(pltgrid, "CC")

        ex, ez, V = solve_2D_E(rho1, rho2, h, A_loc, B_loc)

        ex, ez = Pc * ex, Pc * ez
        Vplt = (Pc * V).reshape(x.size, z.size, order="F")
        fudgeFactor = (
            layer_potentials(
                rho1,
                rho2,
                h,
                A_loc,
                B_loc,
                np.c_[x[match_point], 0.0, 0.0],
            )
            / Vplt[match_point, 0]
        )

        # ex, ez, _ = layer_E(rho1, rho2, h, np.r_[A, 0., 0.], np.r_[B, 0., 0.], np.c_[pltgrid, np.zeros_like(pltgrid[:, 0])])
        ex = fudgeFactor * ex.reshape(x.size, z.size, order="F")
        ez = fudgeFactor * ez.reshape(x.size, z.size, order="F")
        e = np.sqrt(ex ** 2.0 + ez ** 2.0)

        cb = ax[1].pcolor(xplt, zplt, e, cmap="viridis", norm=LogNorm())
        ax[1].plot(
            [xplt.min(), xplt.max()],
            -h * np.r_[1.0, 1],
            color=[0.5, 0.5, 0.5],
            linewidth=1.5,
        )

        #if dipole_source:
        #    clim = np.r_[3e-3, 1e1]

        ax[1].streamplot(
            x,
            z,
            ex.T,
            ez.T,
            color="k",
            linewidth=2
            * (np.log(e.T) - np.log(e).min())
            / (np.log(e).max() - np.log(e).min()),
        )

        clabel = "Electric Field (V/m)"

    elif imgplt == "J":

        Pc = mesh.getInterpolationMat(pltgrid, "CC")

        Jx, Jz, V = solve_2D_J(rho1, rho2, h, A_loc, B_loc)

        Jx, Jz = Pc * Jx, Pc * Jz

        Vplt = (Pc * V).reshape(x.size, z.size, order="F")
        fudgeFactor = (
            layer_potentials(
                rho1,
                rho2,
                h,
                A_loc,
                B_loc,
                np.c_[x[match_point], 0.0, 0.0],
            )
            / Vplt[match_point, 0]
        )

        Jx = fudgeFactor * Jx.reshape(x.size, z.size, order="F")
        Jz = fudgeFactor * Jz.reshape(x.size, z.size, order="F")

        J = np.sqrt(Jx ** 2.0 + Jz ** 2.0)

        cb = ax[1].pcolor(xplt, zplt, J, cmap="viridis", norm=LogNorm())
        ax[1].plot(
            [xplt.min(), xplt.max()],
            -h * np.r_[1.0, 1],
            color=[0.5, 0.5, 0.5],
            linewidth=1.5,
        )
        ax[1].streamplot(
            x,
            z,
            Jx.T,
            Jz.T,
            color="k",
            linewidth=2
            * (np.log(J.T) - np.log(J).min())
            / (np.log(J).max() - np.log(J).min()),
        )
        ax[1].set_ylabel("z (m)", fontsize=16)

        #if dipole_source:
        #    clim = np.r_[3e-5, 3e-2]
        clabel = "Current Density (A/m$^2$)"

    ax[1].set_xlim([x.min(), x.max()])
    ax[1].set_ylim([z.min(), 6.0])
    ax[1].set_ylabel("z (m)", fontsize=16)
    cbar_ax = fig.add_axes([1.0, 0.08, 0.04, 0.4])
    plt.colorbar(cb, cax=cbar_ax, label=clabel)
    if "clim" in locals():
        cb.set_clim(clim)
    ax[1].set_xlabel("x(m)", fontsize=16)

    xytextA1 = (A - 0.75, 2.5)
    ax[1].plot(A, 1.0, marker="v", color="red", markersize=markersize)
    ax[1].annotate("A", xy=xytextA1, xytext=xytextA1, fontsize=fontsize)

    if dipole_source:
        xytextB1 = (B - 0.75, 2.5)
        ax[1].plot(B, 1.0, marker="v", color="blue", markersize=markersize)
        ax[1].annotate("B", xy=xytextB1, xytext=xytextB1, fontsize=fontsize)

    xytextM1 = (M - 0.75, 2.5)
    ax[1].plot(M, 1.0, marker="^", color="yellow", markersize=markersize)
    ax[1].annotate("M", xy=xytextM1, xytext=xytextM1, fontsize=fontsize)

    if dipole_receiver:
        xytextN1 = (N - 0.75, 2.5)
        ax[1].plot(N, 1.0, marker="^", color="green", markersize=markersize)
        ax[1].annotate("N", xy=xytextN1, xytext=xytextN1, fontsize=fontsize)

    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_layer_potentials_app():
    def plot_layer_potentials_interact(survey, A, B, M, N, rho1, rho2, h, Plot):
        return plot_layer_potentials(survey, rho1, rho2, h, A, B, M, N, Plot)

    app = widgetify(
        plot_layer_potentials_interact,
        survey=ToggleButtons(
            options=["Dipole-Dipole", "Dipole-Pole", "Pole-Dipole", "Pole-Pole"],
            value="Dipole-Dipole",
        ),
        A=FloatSlider(
            min=-40.0, max=40.0, step=1.0, value=-30.0, continuous_update=False
        ),
        B=FloatSlider(
            min=-40.0, max=40.0, step=1.0, value=30.0, continuous_update=False
        ),
        M=FloatSlider(
            min=-40.0, max=40.0, step=1.0, value=-10.0, continuous_update=False
        ),
        N=FloatSlider(
            min=-40.0, max=40.0, step=1.0, value=10.0, continuous_update=False
        ),
        rho1=FloatText(
            min=rhomin,
            max=rhomax,
            value=500.0,
            continuous_update=False,
            description="$\\rho_1$",
        ),
        rho2=FloatText(
            min=rhomin,
            max=rhomax,
            value=500.0,
            continuous_update=False,
            description="$\\rho_2$",
        ),
        h=FloatSlider(min=0.0, max=40.0, step=1.0, value=5.0, continuous_update=False),
        Plot=ToggleButtons(options=["Model", "Potential", "E", "J"], value="Model"),
    )
    return app


# From geoscilabs base
class MyApp(ipywidgets.Box):
    def __init__(self, widgets, kwargs):
        self._kwargs = kwargs
        self._widgets = widgets
        super(MyApp, self).__init__(widgets)
        self.layout.display = "flex"
        self.layout.flex_flow = "column"
        self.layout.align_items = "stretch"

    @property
    def kwargs(self):
        return dict(
            [
                (key, val.value)
                for key, val in self._kwargs.items()
                if isinstance(val, (ipywidgets.widget.Widget, ipywidgets.fixed))
            ]
        )


def widgetify(fun, layout=None, manual=False, **kwargs):

    f = fun

    if manual:
        app = ipywidgets.interact_manual(f, **kwargs)
        app = app.widget
    else:
        app = ipywidgets.interactive(f, **kwargs)

    # if layout is None:
    # TODO: add support for changing layouts
    w = MyApp(app.children, kwargs)

    f.widget = w
    # defaults =  #dict([(key, val.value) for key, val in kwargs.iteritems() if isinstance(val, Widget)])
    app.update()
    # app.on_displayed(f(**(w.kwargs)))

    return w


if __name__ == "__main__":
    rho1, rho2 = rhomin, rhomax
    h = 5.0
    A, B = -30.0, 30.0
    M, N = -10.0, 10.0
    Plot = "e"
    plot_layer_potentials(rho1, rho2, h, A, B, M, N, Plot)
