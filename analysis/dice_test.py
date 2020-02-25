from scipy.stats import norm
import math
def dice_test(stat, alfa, mode_speed):
    # add parametrs for normal distribution
    if mode_speed == "plt_6":
        critical_const = norm.ppf(alfa, loc=0.2700502, scale=math.sqrt(0.0002832429))
        p_value = norm.cdf(stat, loc=0.2700502, scale=math.sqrt(0.0002832429))
    if mode_speed == "plt_13.5":
        critical_const = norm.ppf(alfa, loc=0.27328, scale=math.sqrt(0.0009519578))
        p_value = norm.cdf(stat, loc=0.27328, scale=math.sqrt(0.0009519578))
    if mode_speed == "plt_21":
        critical_const = norm.ppf(alfa,  loc=0.4341689, scale=math.sqrt(0.001787848))
        p_value = norm.cdf(stat, loc=0.4341689, scale=math.sqrt(0.001787848))
    if mode_speed == "qpz_13.5":
        critical_const = norm.ppf(alfa, loc=0.4889668, scale=math.sqrt(0.00203925))
        p_value = norm.cdf(stat, loc=0.4889668, scale=math.sqrt(0.00203925))
    if mode_speed == "str_13.5":
        critical_const = norm.ppf(alfa, loc=0.4008677, scale=math.sqrt(0.001664502))
        p_value = norm.cdf(stat, loc=0.4008677, scale=math.sqrt(0.001664502))
    if mode_speed == "str_21":
        critical_const = norm.ppf(alfa, loc=0.4732636, scale=math.sqrt(0.002565701))
        p_value = norm.cdf(stat, loc=0.4732636, scale=math.sqrt(0.002565701))
    if mode_speed == "toe_13.5":
        critical_const = norm.ppf(alfa)
        p_value = norm.cdf(stat)
    if mode_speed == "air_13.5":
        critical_const = norm.ppf(alfa)
        p_value = norm.cdf(stat)

    return critical_const, p_value