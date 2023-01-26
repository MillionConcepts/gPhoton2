import warnings
from functools import partial
from multiprocessing import Pool

from cv2 import circle, FILLED
from cytoolz import valfilter
import numpy as np
from more_itertools import divide
from scipy.optimize import fsolve
from scipy.special import lambertw
import sympy as sp


# noinspection PyPep8Naming
def unit_flux_lambda(threshold, fwhm):
    A, sigma, r, r0, tau_f = sp.symbols('A,sigma,r,r0,tau_f', positive=True)
    sym_gauss2d = A * sp.exp(-(r ** 2 / sigma ** 2))
    unit_flux = sp.Integral(sym_gauss2d, (r, r0, r0 + 1)).doit()
    unit_flux_threshold = (unit_flux - tau_f).subs(
        {sigma: fwhm / 1.665, tau_f: threshold}
    )
    return sp.lambdify((r0, A), unit_flux_threshold)


def peaksolver(peaks, unit_flux_eqn, threshold, fwhm, min_radius):
    # noinspection PyPep8Naming
    sigma, tau_f, r0, A = sp.symbols('sigma,tau_f,r0,A', positive=True)
    unit_flux_threshold = (unit_flux_eqn - tau_f).subs(
        {sigma: fwhm / 1.665, tau_f: threshold}
    )
    ufl = sp.lambdify((r0, A), unit_flux_threshold)
    equations = [partial(ufl, A=cps) for cps in peaks]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # noinspection PyTypeChecker
        solutions = [fsolve(equation, 6) for equation in equations]
    solutions = np.array([s[0] for s in solutions])
    solutions = np.where(solutions < min_radius, min_radius, solutions)
    return solutions


def cps2radii(peaks, threshold, fwhm, min_radius, threads=None):
    A, sigma, r, r0, tau_f = sp.symbols('A,sigma,r,r0,tau_f', positive=True)
    sym_gauss2d = A * sp.exp(-(r ** 2 / sigma ** 2))
    unit_flux = sp.Integral(sym_gauss2d, (r, r0, r0 + 1)).doit()
    if threads is not None:
        divisions = divide(threads, peaks)
        results = {}
        pool = Pool(threads)
        for ix, division in enumerate(divisions):
            results[ix] = pool.apply_async(
                peaksolver,
                (tuple(division), unit_flux, threshold, fwhm, min_radius)
            )
        pool.close()
        pool.join()
        results = [results[i].get() for i in range(threads)]
        pool.terminate()
    else:
        results = peaksolver(peaks, unit_flux, threshold, fwhm, min_radius)
    return np.hstack(results)


def make_background_mask(
    source_table,
    shape,
    threshold=0.004,
    fwhm=16,
    min_radius=2,
    peak_col='peak',
    threads=None,
    **_  # note: sloppy

):
    source_table['bg_radius'] = cps2radii(
        source_table[peak_col].to_numpy(), threshold, fwhm, min_radius, threads
    )
    positions = source_table[
        ['xcentroid', 'ycentroid', 'bg_radius']
    ].round().astype(np.int32)
    bg_canvas = np.zeros(shape, dtype=np.uint8)
    for _, row in positions.iterrows():
        circle(
            bg_canvas,
            (row['xcentroid'], row['ycentroid']), row['bg_radius'],
            1,
            FILLED
        )
    return bg_canvas


# NOTE: below this line is likely deprecated
def derive_flux_expressions():
    infinity = sp.sympify('oo')
    A, n, sigma, r, r0, tau_b, tau_f, epsilon = sp.symbols(
        'A,n,sigma,r,r0,tau_b,tau_f, epsilon', positive=True
    )

    sym_gauss2d = A * sp.exp(-(r ** 2 / sigma ** 2))
    total_flux = sp.Integral(sym_gauss2d, (r, 0, infinity)).doit()
    center_flux = sp.Integral(sym_gauss2d, (r, 0, r0)).doit()
    flux_threshold_radius = sp.solve(
        total_flux - center_flux - tau_f, r0
    )[0]
    annulus_flux = sp.Integral(
        sym_gauss2d, (r, r0, r0 + epsilon)
    ).doit()
    annulus_area = sp.pi * ((r0 + epsilon) ** 2 - r0 ** 2)
    annulus_surface_brightness = annulus_flux / annulus_area
    instantaneous_brightness = sp.limit(
        annulus_surface_brightness, epsilon, 0
    )
    brightness_threshold_radius = sp.solve(
        instantaneous_brightness - tau_b, r0
    )[0]
    return valfilter(
        lambda f: isinstance(f, sp.core.expr.Expr), locals()
    )


# noinspection PyPep8Naming
def numeric_brightness_threshold(A, fwhm, tau_f):
    """
    A, fwhm: height / fwhm of symmetric 2D gaussian.
    tau_f: threshold surface brightness (in same units as A).
    returns: radius (in same units as fwhm).
    """
    sigma = fwhm / 1.655
    return (
        A
        * np.exp(
            -1/2
            * lambertw((1/2)*A**2/(np.pi**2*sigma**2*tau_f**2))
        )
        / (np.pi*tau_f)
        / 2
    ).real


