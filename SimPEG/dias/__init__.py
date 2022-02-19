try:
    # from ..dias import simulation
    from ..dias.electromagnetics.static.resistivity import simulation
    # from ..dias.electromagnetics.static.resistivity import receivers
    from ..dias.electromagnetics.static.induced_polarization import simulation
    from ..dias.electromagnetics.frequency_domain import simulation
    # import SimPEG.dias.potential_fields.base
    # import SimPEG.dias.potential_fields.gravity.simulation
    # import SimPEG.dias.potential_fields.magnetics.simulation
    # from ..dias import objective_function
    from ..dias import data_misfit
    from ..dias import optimization
    from ..dias import inverse_problem
    from ..dias import worker_utils

except ImportError as err:
    print("unable to load dias operations")
    print(err)
