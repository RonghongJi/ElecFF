import math

import numpy as onp

import jax
import jax.numpy as jnp


def _j_0(r: jnp.ndarray) -> jnp.ndarray:
    """Order-0 spherical Bessel function of the first kind.

    The evaluation is delegated to jax.numpy.sinc. The condition r >= 0 is
    not assessed.

    Args:
        r: The radial coordinate.

    Returns:
        The values of the order-0 spherical Bessel function of the first kind.
    """
    return jnp.sinc(r / jnp.pi)


def _j_1(r: jnp.ndarray) -> jnp.ndarray:
    """Order-1 spherical Bessel function of the first kind.

    The function is built on the basis of calls to existing jax.numpy code. The
    condition r >= 0 is not assessed.

    Args:
        r: The radial coordinate.

    Returns:
        The values of the order-1 spherical Bessel function of the first kind.
    """
    return (jnp.sinc(r / jnp.pi) - jnp.cos(r)) / r


def _calc_starting_order_Cai(order: int, r):
    """Use Cai's recipe to compute a starting point for his recurrence.

    When the method of Computer Physics Communications 182 (2011) 664,
    doi:10.1016/j.cpc.2010.11.019 is used to calculate j_l, an starting
    order is required for the descending recurrence. The original
    prescription works for any complex number, but this implementation
    only works for real arguments.

    Args:
        order: The order of the spherical Bessel function to be evaluated.
        r: A single value of the argument to the spherical bessel function.

    Returns:
        An estimate of the order at which to start Cai's descending recurrence.
    """
    cais_m = int(onp.floor(1.83 * r**0.91 + 9.))
    lower_bound = max(order + 1, cais_m)
    # This bound should depend on the kind of floating point used, but that
    # is not a critical point for our narrow use case.
    upper_bound = int(onp.floor(235. + 50. * onp.sqrt(r)))
    return min(lower_bound, upper_bound)


def create_j_l(order: int, dtype: onp.dtype = onp.float32):
    """Generate the l-th spherical Bessel function of the first kind.

    This implementation is only intended for real arguments, and the focus is
    on compatibility with JAX. The calculation is handled by two different
    recurrence strategies, depending on whether the argument is less than or
    greater than the order.

    Args:
        order: The order of the spherical Bessel function, which cannot be
            negative.
        dtype: A floating-point data type on which the function is intended to
           operate.

    Returns:
        A callable that takes radial coordinates and returns the values of
            the selected Bessel function.

    Raises:
        ValueError: if "order" is negative.
    """
    if order < 0:
        raise ValueError("the order of the function cannot be negative")
    elif order == 0:
        return _j_0
    elif order == 1:
        return _j_1

    def j_l_upward(r: jnp.ndarray) -> jnp.ndarray:
        """Order-l spherical Bessel function of the first kind with derivative.

        This implementation, intended for r>l, is based on ascendent recurrence
        and adapted from SciPy itself. The condition r >= 0 is not assessed.

        Args:
            r: The radial coordinate.

        Returns:
            The values of the order-l spherical Bessel function of the first
                kind and the values of its derivative, in a tuple.
        """
        order_0 = _j_0(r)
        order_1 = _j_1(r)
        for i in range(order - 1):
            temp = (2 * i + 3) / r * order_1 - order_0
            order_0, order_1 = order_1, temp
        derivative = order_0 - (order + 1) * order_1 / r
        return (order_1, derivative)

    starting_order = _calc_starting_order_Cai(order, order)
    # Cai's original prescription is 1e-305, which works for double
    # precision but causes huge problems with the derivative. eps**2 * r
    # seems to work well, since smaller r require a smaller value.
    initial_prefactor = onp.finfo(dtype).eps**2

    def j_l_Cai(r: jnp.ndarray) -> jnp.ndarray:
        """Order-l spherical Bessel function of the first kind with derivative.

        This implementation is based on the algorithm developed by L.-W. Cai
        and published in Computer Physics Communications 182 (2011) 664,
        doi:10.1016/j.cpc.2010.11.019 . Whereas Cai's focus is on complex
        numbers, this implementation is intended only for use with real
        arguments in JAX-based differentiable programs. Moreover, the potential
        arguments are assumed to be bounded from above by the order of the
        function.The condition r >= 0 is not assessed.

        Args:
            r: The radial coordinate.

        Returns:
            The values of the order-l spherical Bessel function of the first
                kind and the values of its derivative, in a tuple.
        """
        plus_1 = jnp.zeros_like(r)
        temp = initial_prefactor * r * jnp.ones_like(r)

        # Iterate from the starting order to the desired one.
        for i in range(starting_order - order):
            minus_1 = (2 * (starting_order - i) + 1) * temp / r - plus_1
            plus_1, temp = temp, minus_1

        unnormalized = minus_1
        unnormalized_derivative = order * unnormalized / r - plus_1

        # And the continue all the way to zero to recover the right
        # normalization.
        for i in range(order):
            minus_1 = (2 * (order - i) + 1) * temp / r - plus_1
            plus_1, temp = temp, minus_1

        # Generally speaking, we use the explicit form of j_0 to obtain the
        # right normalization for our function, but close to the zeros of j_0
        # it is more convenient to use j_1 instead.
        order_0 = _j_0(r)
        order_1 = _j_1(r)
        prefactor = jnp.where(
            jnp.fabs(order_0) >= jnp.fabs(order_1), order_0 / minus_1,
            order_1 / plus_1)
        return (prefactor * unnormalized, prefactor * unnormalized_derivative)

    @jax.custom_jvp
    def j_l(r: jnp.ndarray) -> jnp.ndarray:
        """Order-l spherical Bessel function of the first kind.

        This function just dispatches each argument to the right implementation
        based on whether r is lower than the order of the function or not. The
        condition r >= 0 is not assessed.

        Args:
            r: The radial coordinate.

        Returns:
            The values of the order-l spherical Bessel function of the first
                kind.
        """
        r = jnp.asarray(r)
        condition = r < order
        return condition * j_l_Cai(
            r.clip(max=order))[0] + (1. - condition) * j_l_upward(r)[0]

    # Since the derivatives of the spherical Bessel functions are inexpensive
    # to calculate, it pays off to derine a custon jvp rule for them.
    @j_l.defjvp
    def j_l_jvp(primals, tangents):
        r, = primals
        r_dot, = tangents
        Cai, Cai_dot = j_l_Cai(r.clip(max=order))
        upward, upward_dot = j_l_upward(r)
        condition = r < order
        primal_out = condition * Cai + (1. - condition) * upward
        tangent_out = (condition * Cai_dot +
                       (1. - condition) * upward_dot) * r_dot
        return primal_out, tangent_out

    return j_l
