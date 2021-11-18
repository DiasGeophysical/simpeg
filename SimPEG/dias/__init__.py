try:
    import SimPEG.dias.simulation
    import SimPEG.dias.electromagnetics.static.resistivity.simulation
    import SimPEG.dias.electromagnetics.static.resistivity.receivers
    import SimPEG.dias.electromagnetics.static.induced_polarization.simulation
    import SimPEG.dias.electromagnetics.frequency_domain.simulation
    # import SimPEG.dias.potential_fields.base
    # import SimPEG.dias.potential_fields.gravity.simulation
    # import SimPEG.dias.potential_fields.magnetics.simulation
    import SimPEG.dias.objective_function
    import SimPEG.dias.data_misfit
    import SimPEG.dias.optimization
    import SimPEG.dias.inverse_problem

except ImportError as err:
    print("unable to load dias operations")
    print(err)
