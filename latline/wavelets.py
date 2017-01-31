def even_wavelet(s, x, y):
    psi = (1 - 2 * ((s - x) / y) ** 2) / \
          ((1 + ((s - x) / y) ** 2) ** (5. / 2))
    return psi


def odd_wavelet(s, x, y):
    psi = (-3 * (s - x) / y) / \
          ((1 + ((s - x) / y) ** 2) ** (5. / 2))
    return psi