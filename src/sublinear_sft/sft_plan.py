import numpy as np
import sympy


class SFTPlan:
    """Object used for constructing sparse Fourier approximation of a signal.

        The execution of this plan uses Algorithm 3: Fourier Approximate 2 from
        [Iwen, 2013] to construct a $2 * k$ sparse approximation of the Fourier
        coefficients of a periodic function on $[0, 1]$ contained in the
        bandwidth $N$. Additionally, the plan precomputes necessary sampling
        nodes for the desired sparsity and bandwidth (and optional parameters)
        so that it may be reused for different signals which have been sampled
        at the necessary points.

        Attributes:
            N (int): Bandwidth.
            k (int): Sparsity level.
            C (int): Scaling parameter for number of major primes.
                See SFTPlan._major_primes.

        """

    def __init__(self, N: int, k: int, C: int | None = None):
        """Generate an SFTPlan.

        SFTPlan objects hold the initial parameters of the problem, and are
        able to execute a deterministic or randomized SFT on a signal

        Args:
            N (int): Bandwidth.
            k (int): Sparsity level.
            C (int): Scaling parameter for number of major primes.
                See SFTPlan._major_primes. If `None`, will be determined
                automatically based on theory and choice of random or
                deterministic algorithm.
        """
        self.N = N
        self.k = k
        if not C:
            self.C = 14  # TODO: change for deterministic vs random
        self._prime_list = sympy.ntheory.generate.Sieve()

    def _major_primes(self) -> np.ndarray:
        """Constructs major primes for measurement matrix.

        The major primes are calulated so that the first major prime s[0] has a
        sequence of powers of primes smaller than s[0] such that their product
        is at least N / s[0]. The remaining major primes are those sequentially
        following the above calculated s[0]. We determine K = floor(C * k *
        floor(log(N) / log(s[0]))) + 1 total major primes as directed in [Iwen,
        2013].

        Returns:
            np.ndarray of major primes
        """
        num_minor_primes = 0
        first_prime_found = False
        while not first_prime_found:
            num_minor_primes = num_minor_primes + 1

            s0 = self._prime_list[num_minor_primes + 1]
            t_base = self._prime_list[1:(num_minor_primes + 1)]

            # Ensure that for each minor prime base tBase[i],
            #   tBase[i]^alpha[i] < s0 < tBase[i]^(alpha[i] + 1)

            alpha = np.floor(np.log(s0) / np.log(t_base))
            if np.prod(t_base ** alpha) >= self.N / s0:
                first_prime_found = True

        # Set remaining primes

        num_major_primes = int(
            np.floor(
                self.C * self.k * np.floor(np.log(self.N) / np.log(s0))
            ) + 1
        )

        # TODO: Add optional prime shift

        return np.array(
            self._prime_list[(num_minor_primes + 1):(num_minor_primes +
                                                     num_major_primes + 1)]
        )
