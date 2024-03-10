


def distribution_constant(log_lc, lower_bound, upper_bound, lower_return, upper_return, mid_return):
    if log_lc <= lower_bound:
        return lower_return
    elif log_lc >= upper_bound:
        return upper_return
    else:
        return mid_return(log_lc)

def alpha(sat_type, log_lc):
    if sat_type == "ROCKET_BODY":
        return distribution_constant(log_lc, -1.4, 0.0, 1.0, 0.5,
                                     lambda log_lc: 1.0 - 0.3571 * (log_lc + 1.4))
    else:
        return distribution_constant(log_lc, -1.95, 0.55, 0.0, 1.0,
                                     lambda log_lc: 0.3 + 0.4 * (log_lc + 1.2))

def mu_1(sat_type, log_lc):
    if sat_type == "ROCKET_BODY":
        return distribution_constant(log_lc, -0.5, 0.0, -0.45, -0.9,
                                     lambda log_lc: -0.45 - 0.9 * (log_lc + 0.5))
    else:
        return distribution_constant(log_lc, -1.1, 0.0, -0.6, -0.95,
                                     lambda log_lc: -0.6 - 0.318 * (log_lc + 1.1))

def sigma_1(sat_type, log_lc):
    if sat_type == "ROCKET_BODY":
        return 0.55
    else:
        return distribution_constant(log_lc, -1.3, -0.3, 0.1, 0.3,
                                     lambda log_lc: 0.1 + 0.2 * (log_lc + 1.3))

def mu_2(sat_type, log_lc):
    if sat_type == "ROCKET_BODY":
        return -0.9
    else:
        return distribution_constant(log_lc, -1.3, -0.3, 0.1, 0.3,
                                     lambda log_lc: 0.1 + 0.2 * (log_lc + 1.3))

def sigma_2(sat_type, log_lc):
    if sat_type == "ROCKET_BODY":
        return distribution_constant(log_lc, -1.0, 0.1, 0.28, 0.1,
                                     lambda log_lc: 0.28 - 0.1636 * (log_lc + 1.0))
    else:
        return distribution_constant(log_lc, -0.5, -0.3, 0.5, 0.3,
                                     lambda log_lc: 0.5 - (log_lc + 0.5))
def mu_soc(log_lc):
    return distribution_constant(log_lc, -1.75, -1.25, -0.3, -1.0,
                                 lambda log_lc: -0.3 - 1.4 * (log_lc + 1.75))
def sigma_soc(log_lc):
    return 0.2 if log_lc <= -3.5 else 0.2 + 0.1333 * (log_lc + 3.5)

