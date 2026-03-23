"""
Python model 'LotkaVolterra.py'
Translated using PySD
"""

from pathlib import Path
import numpy as np

from pysd.py_backend.statefuls import Integ
from pysd import Component

__pysd_version__ = "3.14.3"

__data = {"scope": None, "time": lambda: 0}

_root = Path(__file__).parent


component = Component()

#######################################################################
#                          CONTROL VARIABLES                          #
#######################################################################

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 10000,
    "time_step": lambda: 0.001,
    "saveper": lambda: time_step(),
}


def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]


@component.add(name="Time")
def time():
    """
    Current time of the model.
    """
    return __data["time"]()


@component.add(
    name="FINAL TIME", units="Day", comp_type="Constant", comp_subtype="Normal"
)
def final_time():
    """
    The final time for the simulation.
    """
    return __data["time"].final_time()


@component.add(
    name="INITIAL TIME", units="Day", comp_type="Constant", comp_subtype="Normal"
)
def initial_time():
    """
    The initial time for the simulation.
    """
    return __data["time"].initial_time()


@component.add(
    name="SAVEPER",
    units="Day",
    limits=(0.0, np.nan),
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time_step": 1},
)
def saveper():
    """
    The frequency with which output is stored.
    """
    return __data["time"].saveper()


@component.add(
    name="TIME STEP",
    units="Day",
    limits=(0.0, np.nan),
    comp_type="Constant",
    comp_subtype="Normal",
)
def time_step():
    """
    The time step for the simulation.
    """
    return __data["time"].time_step()


#######################################################################
#                           MODEL VARIABLES                           #
#######################################################################


@component.add(
    name="Conejos",
    units="conejos",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_conejos": 1},
    other_deps={
        "_integ_conejos": {
            "initial": {},
            "step": {"nac_conejos": 1, "muert_conejos": 1},
        }
    },
)
def conejos():
    return _integ_conejos()


_integ_conejos = Integ(
    lambda: nac_conejos() - muert_conejos(), lambda: 100, "_integ_conejos"
)


@component.add(
    name="Lobos",
    units="lobos",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_lobos": 1},
    other_deps={
        "_integ_lobos": {"initial": {}, "step": {"nac_lobos": 1, "muert_lobos": 1}}
    },
)
def lobos():
    return _integ_lobos()


_integ_lobos = Integ(lambda: nac_lobos() - muert_lobos(), lambda: 25, "_integ_lobos")


@component.add(
    name="Muert Conejos",
    units="conejos/Day",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"conejos": 1, "lobos": 1, "tasa_muert_conejos": 1},
)
def muert_conejos():
    return conejos() * lobos() * tasa_muert_conejos()


@component.add(
    name="Muert Lobos",
    units="lobos/Day",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"lobos": 1, "tasa_muert_lobos": 1},
)
def muert_lobos():
    return lobos() * tasa_muert_lobos()


@component.add(
    name="Nac Conejos",
    units="conejos/Day",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"conejos": 1, "tasa_nac_conejos": 1},
)
def nac_conejos():
    return conejos() * tasa_nac_conejos()


@component.add(
    name="Nac Lobos",
    units="lobos/Day",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"conejos": 1, "lobos": 1, "tasa_nac_lobos": 1},
)
def nac_lobos():
    return conejos() * lobos() * tasa_nac_lobos()


@component.add(
    name="Tasa muert Conejos",
    units="1/(lobos*Day)",
    comp_type="Constant",
    comp_subtype="Normal",
)
def tasa_muert_conejos():
    return 0.0025


@component.add(
    name="Tasa muert Lobos", units="1/Day", comp_type="Constant", comp_subtype="Normal"
)
def tasa_muert_lobos():
    return 0.04


@component.add(
    name="Tasa nac Conejos", units="1/Day", comp_type="Constant", comp_subtype="Normal"
)
def tasa_nac_conejos():
    return 0.01


@component.add(
    name="Tasa nac Lobos",
    units="1/(conejos * Day)",
    comp_type="Constant",
    comp_subtype="Normal",
)
def tasa_nac_lobos():
    return 0.002
