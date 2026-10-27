from functools import cache

import numpy as np
import scipy as sp
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

    @property
    @cache
    def major_primes(self) -> np.ndarray:
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

    @property
    @cache
    def minor_primes(self) -> dict:
        """Constructs minor "primes" for the given major prime.
        The minor primes are actually powers of primes of the form
        t(i) = tBase(i) ^ alpha(i), which satisfy prod(t) >= N / majorPrime and
        are relatively prime to majorPrime. The minor primes are chosen
        heuristically so that the the minor primes stay close to each other, do
        not grow much larger than the associated majorPrime, and have a small
        sum. The specific rule initially appeared in the implementation of
        DMSFT by Ruochuan Zhang, see
        https://sourceforge.net/projects/aafftannarborfa/.

        If majorPrime is a vector, t will be a cell array of vectors of minor
        primes, one which satisfies the above conditions for each element of
        majorPrime. For consistency, if majorPrime has only one element, t will
        still be a cell array containing a vector of minor primes.

        The following example suggests the usage of powers of primes for minor
        primes: Suppose that 300 = N / s for some major prime s. If we use
        successive primes for the minor primes, we would obtain
        2 * 3 * 5 * 7 * 11 > 300,
        meaning the minor primes sum to 28.
        However, if we use powers of primes,
        we can have 3^2 * 5 * 7 > 300,
        meaning the minor primes sum to only 21 and we only have three of them.
        This reduces the number of rows in the measurement matrix as well as
        the sizes of the FFTs used to generate it.

        Returns:
            dict of np.ndarray of minor primes with keys being major primes
        """

        result = dict()
        for i in range(len(self.major_primes)):
            t_base = np.array([self._prime_list[1]])
            alpha = np.array([1])
            minor_prime_product = np.prod(t_base ** alpha)
            # Keep track of the next prime to use as minor prime.
            next_prime = self._prime_list[2]
            while minor_prime_product < self.N / self.major_primes[i]:
                # To keep the minor primes close, we will only add a new prime
                # for a base if it is smaller than (t_base .^ alpha) .* (t_base
                # - 1).  Otherwise, we will bump up the exponent corresponding
                # to the smallest of these values. This ensures that as the
                # loop progresses, the minorPrimeProduct grows slowly towards N
                # / majorPrime[i].
                test_array = t_base ** alpha * (t_base - 1)
                idx = test_array.argmin()
                minimal_test_value = test_array[idx]
                if next_prime < minimal_test_value:
                    t_base = np.append(t_base, [next_prime])
                    alpha = np.append(alpha, [1])
                    if (self._prime_list[len(t_base) + 1]
                            >= self.major_primes[i]):
                        next_prime = self._prime_list[len(t_base) + 2]
                    else:
                        next_prime = self._prime_list[len(t_base) + 2]
                else:
                    # Increase exponent
                    alpha[idx] = alpha[idx] + 1
                minor_prime_product = np.prod(t_base ** alpha)
            result[self.major_primes[i]] = t_base ** alpha

        return result

    @property
    @cache
    def major_minor_prime_product(self) -> np.ndarray:
        return np.concatenate(
            (
                self.major_primes,
                np.concatenate([
                    major_prime * self.minor_primes[major_prime]
                    for major_prime in self.major_primes
                ])
            )
        )

    @property
    @cache
    def major_minor_prime_product_sum(self) -> np.ndarray:
        return np.concatenate(
            ([0], self.major_minor_prime_product.cumsum())
        )

    def _general_fast_multiply(self, samples: np.ndarray):
        """Product of large measurement matrix and Fourier transforms

        Calculates the product (denoted in [Iwen, 2013] G_{\lambda, K} \tilde Phi A)
        of an extended row tensor product measurement matrix for two sets of
        relatively prime integers with the Fourier transform of a signal. The
        computation is performed using a series of smaller FFTs of sizes
        s(j) * t(i).

         Note: On the indexing scheme for product.
            The row tensor product measurement matrix is such that the first
            sum(s) rows are those corresponding to the original measurement
            matrix for only the set s. Thus, letting sSum = cumsum([0, s]), the
            row r_{j, h} corresponding to prime s(j) and residue h (mod s(j))
            in this original measurement matrix and therefore 'product' is
            found at index sSum(j) + h + 1.

            For the remaining rows, we use the notation \bar r_{i, j, h} to
            refer to rows in the block of the row tensor product matrix
            consisting of rows which are tensor products of r_{j, .} and \tilde
            r_{i, .} where rows r are from the original measurement matrix
            constructed using the values in s as described in the previous
            paragraph and \tilde r are rows from the same form of matrix
            constructed with the values in t{j}. The index h is therefore a
            residue modulo t{j}(i) * s(j). Letting sSum = cumsum([0, s]) as
            above, tSum{j} = cumsum([0, t{j}]), the row \bar r_{i, j, h} in
            'product' is found at index
                sum(s)
                + s(1) * sum(t{1})
                + s(2) * sum(t{2})
                + ... +
                s(j - 1) * sum(t{j - 1}) + s(j) * tSum{j}(i) + h + 1
            When j > 1, the second line above can be calculated with
                dot(s(1:j-1), cellfun(@sum, {t{1:j - 1}}))
            and tSum can be calculated with
                tSum = cellfun(@(x)cumsum([0, x]), t, 'UniformOutput', false).

            Note also that this row corresponds to the row tensor product
                \tilde r_{i, h mod t(i)} .* r_{j, h mod s(j)},
            and so in
            particular, we can index any row tensor product
            \tilde r_{i, b} .* r_{j, k}
            at \bar r_{i, j, h} using the above formula for
            h = self._modular_reconstruction([b, k], [t{j}(i), s(j)]).
        Args:
            samples: Samples of the signal at the sampling nodes for the plan.

        Returns:
            A vector of length sum(s) * (1 + sum(t)) as specified above as the
            extended measurement matrix applied to a large Fourier transform of
            the signal.
        """
        # TODO: Rewrite docs for Python notation and test
        product = np.zeros((sum(self.major_minor_prime_product,)))
        for i in range(len(self.major_minor_prime_product)):
            length = self.major_minor_prime_product[i]
            start = self.major_minor_prime_product_sum[i]
            window = np.arange(start, length)
            # TODO: Check normalization coefficient for scipy's FFT
            product[window] = (1 / length) * sp.fft.fft(samples[window])

        return product
