# Solution to the sampling problem of a parametric curve
# Author: Shakeeb Bin Hasan

import numpy as np
import typing
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def find_t_given_curve_len(
        curve_len_array: np.ndarray,
        ref_curve_len_array: np.ndarray,
        ref_t_array: np.ndarray
):
    """Return the t_array which corresponds to the given curve_len_array
    Args:
        curve_len_array: np.ndarray, numpy array containing curve's length increasing with each segment
        ref_curve_len: np.ndarray, numpy array containing the curve's length sampled over uniform t to interpolate on
        ref_t_array: np.ndarray, numpy array contianing parameters t which are uniformly sampled
    Returns:
        np.ndarray containing values of parameter t corresponding to curve_len_array
    """
    return np.interp(curve_len_array, ref_curve_len_array, ref_t_array)


def find_curve_len(
        x_t: typing.Callable,
        y_t: typing.Callable,
        t_array: np.ndarray,
) -> np.ndarray:
    """Given the parametric functions, returns an np array containing increasing curve length for each segment
    Args:
        x_t: function, defines x(t)
        y_t: function, defines y(t)
        t_array: np.ndarray, parameter values to evaluate x(t) and y(t) on
    Returns:
        curve_len_array: np.ndarray, numpy array containing the curve's length increasing with each segment
    """
    # produce the curve for the given sample
    x, y = x_t(t_array), y_t(t_array)
    segment_len_array = np.zeros(t_array.size)
    segment_len_array[1:] = np.sqrt((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2)
    curve_len_array = np.cumsum(segment_len_array)
    return curve_len_array


def find_upper_limit_T(
        x_t: typing.Callable,
        y_t: typing.Callable,
        target_curve_len: float
) -> typing.Tuple[float, np.ndarray, np.ndarray]:
    """Determine the upper limit T such that t in [0, T] gives a curve length of target_curve_len
    Args:
        x_t: function that determines x(t)
        y_t: function that determines y(t)
        target_curve_len: float, desired length of the curve
    Returns:
        T: float, upper limit T in [0, T] for the parameter t that produces curve with given length
        curve_len_array: np.ndarray, array containing curve's length sampled for t in [0, 1]
        t_array: np.ndarray, array containing uniformly sampled parameter t in [0, 1]
    """
    # create a fine sample of curve with t such that it adequately samples it
    t_array = np.linspace(0, 1, 10000)

    # calculate the curve length
    curve_len_array = find_curve_len(x_t, y_t, t_array)

    # finally we invert the curve_len_array to determine the upper-bound T of the parameter t
    T = np.interp(target_curve_len, curve_len_array, t_array)
    return T, curve_len_array, t_array


def myplot(
        x: np.ndarray,
        y: np.ndarray,
        style: str = None,
        *,
        xlabel: str = None,
        ylabel: str = None,
        title: str = None,
        aspect: str = None
):
    """Function to produce plots according to my preferences"""
    fig, ax = plt.subplots(constrained_layout=True)
    if style is None:
        ax.plot(x, y, linewidth=2)
    else:
        ax.plot(x, y, style, linewidth=2)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if aspect is not None:
        ax.set_aspect(aspect)
    return fig, ax


# I like latex fonts, comment out this segment if it causes error during execution
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 16,
    "font.family": "Helvetica"
})

if __name__ == "__main__":
    #####################################input configuration#####################################

    target_curve_len = 5  # desired length of the curve, it should be achievable for t in [0, 1]
    t_interval = [0, 1]  # interval of the parameter t (defined in the question)
    error_bound = 1e-4  # error bound on the length of the realized curve
    max_num_segments = 10000  # maximum number of curve sampling allowed to prevent the algorithm from running indefinitely

    # define functions x(t) and y(t)
    x_t = lambda t: 20 * t * np.sin(t / 10)  # function x(t)
    y_t = lambda t: 3 * np.exp(t ** 2)  # function y(t)

    ###################################end configuration###################################

    # first we need to determine the parameter T such that t in [0, T] yields a curve whose size is target_curve_len
    T, ref_curve_len_array, ref_t_array = find_upper_limit_T(x_t, y_t, target_curve_len)

    # ensure that the returned curve has the same length as desired in the target_curve_len
    if (ref_curve_len_array[-1] < target_curve_len):
        raise Exception("The desired curve length cannot be reached with t in [0, 1]")

    # now we need to uniformly sample the curve such that its length is within the error bounds

    output_curve_len = 0
    num_segments = 0  # 1 segment means two points and so on
    while (np.abs(output_curve_len - target_curve_len) / target_curve_len > error_bound):
        num_segments += 1
        curve_len_array = np.linspace(0, target_curve_len, num_segments + 1)
        t_array = find_t_given_curve_len(curve_len_array, ref_curve_len_array, ref_t_array)
        curve_len_array = find_curve_len(x_t, y_t, t_array)
        output_curve_len = curve_len_array[-1]
        #         print("segments = {}".format(num_segments))
        if (num_segments > max_num_segments):
            raise Exception("Maximum number of curve segments reached, stopping computation")

    # now that we know T and num_segments, let us produce the output curve
    print("Number of curve segments N = {}, obtained curve length l = {}".format(t_array.size, output_curve_len))
    x_out, y_out = x_t(t_array), y_t(t_array)

    plot_settings = {
        "xlabel": "Segment number $n$",
        "ylabel": "Curve length $l$"
    }

    fig, ax = myplot(range(1, t_array.size), curve_len_array[1:], 'o--', **plot_settings)

    plot_settings["ylabel"] = "Segment length $s$"
    fig, ax = myplot(range(1, t_array.size), curve_len_array[1:] - curve_len_array[:-1], 'o--', **plot_settings)

    plot_settings["xlabel"] = "$x(t)$"
    plot_settings["ylabel"] = "$y(t)$"
    plot_settings["aspect"] = "equal"
    fig, ax = myplot(x_out, y_out, 'o--', **plot_settings)

