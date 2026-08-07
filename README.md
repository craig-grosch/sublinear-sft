# Sublinear Sparse Fourier Transform (SFT)

This is a Python based implementation of the sparse Fourier transform algorithm (Algorithm 3) from [Iwen, 2013].
Given sampling access to a signal, a sparse approximation of its discrete Fourier transform will be calculated. The only parameters necessary are the bandwidth of the signal, as well as the desired sparsity level. Optional parameters are available for faster execution at the cost of potentially less accurate results.


## See also

This code is a reimplementation of [the MATLAB implementation](https://gitlab.com/grosscra/SublinearSparseFourierMATLAB) which itself is a reimplementation of only the nonequispaced sparse Fourier transform used in DMSFT in the [MSU Sparse Fourier Repository](https://sourceforge.net/projects/aafftannarborfa/), originally implemented by Ruochuan Zhang.

## References

[Mark Iwen](https://users.math.msu.edu/users/iwenmark/index.html), **Improved approximation guarantees for sublinear-time Fourier algorithms**, Applied and Computational Harmonic Analysis, 34(1)(2013), pp. 57-82, <https://www.sciencedirect.com/science/article/pii/S1063520312000462>. ([Preprint](https://users.math.msu.edu/users/iwenmark/Papers/Iwen_Improved.pdf) available on the author's webpage.)
