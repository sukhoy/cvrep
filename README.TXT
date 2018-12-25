This project includes two patches for the LIBLINEAR library that make it
possible to parallelize the computation of cross-validation results and make
these results reproducible across platforms. The patches also eliminate the
random fluctuations associated with sharing of the PRNG state in parallel
threads by storing the state in thread-local storage and re-initializing it
before each iteration of cross-validation.

The first patch replaces the PRNG implemented by the standard function rand()
with the SFMT PRNG and stores its state in thread-local storage (TLS). The patch
is built under the assumption that the SFMT source code has been deployed into
the 'SFMT' sub-directory and that SFMT.o has been compiled in that
directory (building SFMT.o is not required on Windows). The patch can be applied
using the following example command line:

$ patch -d [LIBLINEAR source directory] < PRNG/deploy_SFMT.diff


The second patch modifies the cross-validation loop in LIBLINEAR so that it is
parallelized using OpenMP. The patch also inserts the necessary flags into the
build files. It can be applied as follows:

$ patch -d [LIBLINEAR source directory] < OpenMP/deploy_OpenMP_to_CV.diff


Then, build or rebuild LIBLINEAR as described in its documentation (e.g., using
make on Linux or MacOS and nmake on Windows with Visual Studio). If you applied
the OpenMP patch (which parallelizes cross-validation), then the build has to
use a C++ compiler that supports OpenMP. To reproduce the results described in
the paper, please use the following command line:

$ train -c 4 -e 0.1 -v <num_folds> rcv1_train.binary

The example data set rcv1_train can be downloaded from the following link:
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#rcv1.binary

Both patches are released under the 3-clause BSD license (see COPYRIGHT). This
license is similar to the license used by the LIBLINEAR project, the only
difference is the copyright notice.
The license used by LIBLINEAR is stored in LIBLINEAR_LICENSE.TXT



Vladimir Sukhoy and Alexander Stoytchev
