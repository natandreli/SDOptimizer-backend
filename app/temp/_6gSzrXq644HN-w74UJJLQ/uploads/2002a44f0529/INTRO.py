"""
Python model 'INTRO.py'
Translated using PySD
"""

from pathlib import Path

from pysd.py_backend.functions import zidz, if_then_else
from pysd.py_backend.statefuls import Integ
from pysd.py_backend.lookups import HardcodedLookups
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
    "final_time": lambda: 100,
    "time_step": lambda: 0.125,
    "saveper": lambda: 1,
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
    name="FINAL TIME", units="Month", comp_type="Constant", comp_subtype="Normal"
)
def final_time():
    """
    FINAL TIME
    """
    return __data["time"].final_time()


@component.add(
    name="INITIAL TIME", units="Month", comp_type="Constant", comp_subtype="Normal"
)
def initial_time():
    """
    The initial time for the simulation.
    """
    return __data["time"].initial_time()


@component.add(
    name="SAVEPER", units="Month", comp_type="Constant", comp_subtype="Normal"
)
def saveper():
    """
    The frequency with which output is stored.
    """
    return __data["time"].saveper()


@component.add(
    name="TIME STEP", units="Month", comp_type="Constant", comp_subtype="Normal"
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
    name="Average Worker Hours",
    units="Hour/Month",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_average_worker_hours": 1},
    other_deps={
        "_integ_average_worker_hours": {
            "initial": {"normal_workweek": 1},
            "step": {"change_in_hours_worked": 1},
        }
    },
)
def average_worker_hours():
    """
    The average number of hours worked per week
    """
    return _integ_average_worker_hours()


_integ_average_worker_hours = Integ(
    lambda: change_in_hours_worked(),
    lambda: normal_workweek(),
    "_integ_average_worker_hours",
)


@component.add(
    name="change in hours worked",
    units="Hour/(Month*Month)",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "required_worker_hours": 1,
        "average_worker_hours": 1,
        "delay_in_adding_or_removing_overtime_hours": 1,
    },
)
def change_in_hours_worked():
    return (
        required_worker_hours() - average_worker_hours()
    ) / delay_in_adding_or_removing_overtime_hours()


@component.add(
    name="delay in adding or removing overtime hours",
    units="Month",
    comp_type="Constant",
    comp_subtype="Normal",
)
def delay_in_adding_or_removing_overtime_hours():
    return 3


@component.add(
    name="production",
    units="Widget/Month",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "workforce": 1,
        "required_worker_hours": 1,
        "productivity": 1,
        "normal_workweek": 1,
    },
)
def production():
    """
    Total amount produced.
    """
    return workforce() * required_worker_hours() * productivity() / normal_workweek()


@component.add(
    name="required worker hours",
    units="Hour/Month",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "target_production": 1,
        "workforce": 1,
        "productivity": 1,
        "normal_workweek": 1,
    },
)
def required_worker_hours():
    """
    The amount relative to normal that people have to work to produce enough.
    """
    return zidz(target_production(), workforce() * productivity()) * normal_workweek()


@component.add(
    name="net hires",
    units="Person/Month",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"desired_workforce": 1, "workforce": 1, "time_to_hire_or_layoff": 1},
)
def net_hires():
    """
    The net number of people hired (laid off if negative).
    """
    return (desired_workforce() - workforce()) / time_to_hire_or_layoff()


@component.add(
    name="normal workweek",
    units="Hour/Month",
    comp_type="Constant",
    comp_subtype="Normal",
)
def normal_workweek():
    return 40


@component.add(
    name="time to hire or layoff",
    units="Month",
    comp_type="Constant",
    comp_subtype="Normal",
)
def time_to_hire_or_layoff():
    return 6


@component.add(
    name="desired workforce",
    units="Person",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"target_production": 1, "productivity": 1},
)
def desired_workforce():
    """
    The desired number of workers.
    """
    return zidz(target_production(), productivity())


@component.add(
    name="effect of morale on productivity lookup",
    units="Dmnl",
    comp_type="Lookup",
    comp_subtype="Normal",
    depends_on={
        "__lookup__": "_hardcodedlookup_effect_of_morale_on_productivity_lookup"
    },
)
def effect_of_morale_on_productivity_lookup(x, final_subs=None):
    """
    The effect that morale has on the ability to produce
    !
    !
    !
    """
    return _hardcodedlookup_effect_of_morale_on_productivity_lookup(x, final_subs)


_hardcodedlookup_effect_of_morale_on_productivity_lookup = HardcodedLookups(
    [0.0, 1.0, 2.0, 2.63444, 3.29909, 3.97583],
    [0.0, 1.0, 2.0, 2.36842, 2.52632, 2.57895],
    {},
    "interpolate",
    {},
    "_hardcodedlookup_effect_of_morale_on_productivity_lookup",
)


@component.add(
    name="morale",
    units="Dmnl",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"average_worker_hours": 1, "normal_workweek": 1, "morale_lookup": 1},
)
def morale():
    """
    Morale - 1 is normal.
    """
    return morale_lookup(average_worker_hours() / normal_workweek())


@component.add(
    name="morale lookup",
    units="Dmnl",
    comp_type="Lookup",
    comp_subtype="Normal",
    depends_on={"__lookup__": "_hardcodedlookup_morale_lookup"},
)
def morale_lookup(x, final_subs=None):
    """
    Lookup relating the average level of overtime to morale.
    !
    !
    !
    """
    return _hardcodedlookup_morale_lookup(x, final_subs)


_hardcodedlookup_morale_lookup = HardcodedLookups(
    [0.0, 0.326284, 0.495468, 0.682779, 0.839879, 1.0, 1.14804, 1.3716, 1.66767, 2.0],
    [1.5, 1.4386, 1.40351, 1.32018, 1.18421, 1.0, 0.833333, 0.640351, 0.526316, 0.5],
    {},
    "interpolate",
    {},
    "_hardcodedlookup_morale_lookup",
)


@component.add(
    name="normal productivity",
    units="Widget/Month/Person",
    comp_type="Constant",
    comp_subtype="Normal",
)
def normal_productivity():
    """
    The normal amount produced in the absence of morale effects.
    """
    return 1


@component.add(
    name="productivity",
    units="Widget/(Month*Person)",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={
        "normal_productivity": 1,
        "effect_of_morale_on_productivity_lookup": 1,
        "morale": 1,
    },
)
def productivity():
    """
    The amount each peson can produce given morale effects.
    """
    return normal_productivity() * effect_of_morale_on_productivity_lookup(morale())


@component.add(
    name="sales",
    units="Widget/Month",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"time": 1},
)
def sales():
    """
    The total amount sold.
    """
    return if_then_else(time() > 20, lambda: 120, lambda: 100)


@component.add(
    name="target production",
    units="Widget/Month",
    comp_type="Auxiliary",
    comp_subtype="Normal",
    depends_on={"sales": 1},
)
def target_production():
    """
    The target amount produced.
    """
    return sales()


@component.add(
    name="Workforce",
    units="Person",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_workforce": 1},
    other_deps={
        "_integ_workforce": {
            "initial": {"target_production": 1, "normal_productivity": 1},
            "step": {"net_hires": 1},
        }
    },
)
def workforce():
    """
    The number of workers
    """
    return _integ_workforce()


_integ_workforce = Integ(
    lambda: net_hires(),
    lambda: target_production() / normal_productivity(),
    "_integ_workforce",
)
