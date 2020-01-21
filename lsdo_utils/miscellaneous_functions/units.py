import pint


ureg = pint.UnitRegistry()

def units(unit_to, unit_from):
    ureg_from = ureg.__getattr__(unit_from)
    ureg_to = ureg.__getattr__(unit_to)
    return (1. * ureg_from).to(ureg_to).magnitude