import numpy as np

from sublinear_sft.sft_plan import SFTPlan


def test_major_primes():
    plan = SFTPlan(4, 2)
    # First major prime is 3 since
    # 2**1 < 3 and 2**1 = 2 > 4 / 3 = N / s[0]
    assert plan.major_primes[0] == 3

    # Number of major primes is
    # floor(C * k * floor(log_3(N)) + 1)
    # = floor(14 * 2 * 1 + 1) = 29
    assert len(plan.major_primes) == 29

    plan = SFTPlan(10, 2)
    primes = plan.major_primes
    # First major prime is 5 since
    # 2**2 and 3**1 < 5 and 2**2 * 3**1 = 12 > 10 / 5 = N / s[0]

    assert primes[0] == 5
    # Number of major primes is
    # floor(C * k * floor(log_5(N)) + 1)
    # = floor(14 * 2 * 1 + 1) = 29
    assert len(primes) == 29

    plan = SFTPlan(15, 2)
    primes = plan.major_primes
    # First major prime is 5 since
    # 2**2, 3**1 < 5 and 2**2 * 3**1 = 12 > 3 = 15 / 5

    assert primes[0] == 5
    # Number of major primes is
    # floor(C * k * floor(log_5(N)) + 1)
    # = floor(14 * 2 * 1 + 1) = 29
    assert len(primes) == 29

    plan = SFTPlan(84, 2)
    primes = plan.major_primes
    # First major prime is 7 since
    # 2**2, 3**1, 5**1 < 7 and 2**2 * 3**1 * 5**1 = 120 > 12 = 84 / 7

    assert primes[0] == 7
    # Number of major primes is
    # floor(C * k * floor(log_7(N)) + 1)
    # = floor(14 * 2 * 2 + 1) = 57
    assert len(primes) == 57


def test_minor_primes():
    plan = SFTPlan(84, 2)
    major_primes = plan.major_primes
    minor_primes = plan.minor_primes
    assert isinstance(minor_primes, dict)
    assert len(minor_primes) == 57
    for prime in major_primes:
        assert prime in minor_primes.keys()
        assert plan.N / prime <= np.prod(minor_primes[prime])


def test_major_minor_prime_product():
    plan = SFTPlan(84, 2)
    product = plan.major_minor_prime_product
    assert product.shape == (len(plan.major_primes)
                             + sum([len(minor_prime_list)
                                    for minor_prime_list
                                    in plan.minor_primes.values()]),
                             )
    assert np.all(product[:len(plan.major_primes)] == plan.major_primes)


def test_major_minor_prime_product_sum():
    plan = SFTPlan(84, 2)
    product_sum = plan.major_minor_prime_product_sum
    assert product_sum[0] == 0
    assert product_sum.shape == (1 + len(plan.major_minor_prime_product),)
    sum_val = 0
    for idx, val in enumerate(plan.major_minor_prime_product):
        assert product_sum[idx] == sum_val
        sum_val += val


def test_general_fast_multiply():
    pass
