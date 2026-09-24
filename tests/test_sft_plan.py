import numpy as np

from sublinear_sft.sft_plan import SFTPlan


def test_major_primes():
    plan = SFTPlan(4, 2)
    # First major prime is 3 since
    # 2**1 < 5 and 2**1 = 2 > 4 / 3 = N / s[0]

    # Number of major primes is
    # floor(C * k * floor(log_5(N)) + 1)
    # = floor(14 * 2 *  0 + 1) = 1
    assert np.all(plan._major_primes() == np.array([3]))

    plan = SFTPlan(10, 2)
    primes = plan._major_primes()
    # First major prime is 5 since
    # 2**2 and 3**1 < 5 and 2**2 * 3**1 = 12 > N = 10

    assert primes[0] == 5
    # Number of major primes is
    # floor(C * k * floor(log_5(N)) + 1)
    # = floor(14 * 2 * 1 + 1) = 29
    assert len(primes) == 29

    plan = SFTPlan(15, 3)
    primes = plan._major_primes()
    # First major prime is 7 since
    # 2**1, 3**1, and 5**1 < 7 and 2**1 * 3**1 * 5**1 = 30 > N = 15

    assert primes[0] == 7
    # Number of major primes is
    # floor(C * k * floor(log_7(N)) + 1)
    # = floor(14 * 3 * 1 + 1) = 43
    assert len(primes) == 43
