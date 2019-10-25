#!/usr/bin/env python
"""
This script downloads, patches, and builds the source code for the modified
version of LIBLINEAR that implements parallelized cross-validation (CV) without
variability. It also downloads the example data set (RCV1) for testing the code.

It is assumed that the default C++ compiler on the system supports OpenMP.
Another compiler can be set using the 'CC' and 'CXX' environment variables.
"""
from __future__ import print_function
import atexit
import bz2
import os
import os.path as p
import platform
import shutil
import subprocess
import tempfile
try:
    import urllib.request as r  # for Python 3
except ImportError:
    import urllib as r  # for Python 2
import zipfile
import sys


SCRIPT_DIR = p.dirname(__file__)
print(SCRIPT_DIR)
OPENMP_PATCH = p.join(SCRIPT_DIR, 'OpenMP', 'deploy_OpenMP_to_CV.diff')
PRNG_PATCH = p.join(SCRIPT_DIR, 'PRNG', 'deploy_SFMT.diff')



LL_URL = 'https://github.com/cjlin1/liblinear/archive/v221.zip'
LL_FILENAME = 'liblinear-221_orig.zip'

RCV1_URL = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2'
RCV1_FILENAME = 'rcv1_train.binary.bz2'

SFMT_URL = 'https://github.com/MersenneTwister-Lab/SFMT/archive/1.5.1.zip'
SFMT_FILENAME = 'SFMT-1.5.1.zip'

DEST_DIR = 'LIBLINEAR_patched'
SFMT_DIR = p.join(DEST_DIR, 'SFMT')


if __name__ == '__main__':
    msg = "Downloading LIBLINEAR source code from {} to {}".format(LL_URL, LL_FILENAME)
    print("[START] {}".format(msg))
    if not p.exists(LL_FILENAME):
        r.urlretrieve(LL_URL, LL_FILENAME)
    print("[DONE]  {}".format(msg))

    msg = "Downloading the RCV1 dataset from {} to {}".format(RCV1_URL, RCV1_FILENAME)
    print("[START] {}".format(msg))
    if not p.exists(RCV1_FILENAME):
        r.urlretrieve(RCV1_URL, RCV1_FILENAME)
    print("[DONE]  {}".format(msg))

    msg = "Downloading SFMT source code from {} to {}".format(SFMT_URL, SFMT_FILENAME)
    print("[START] {}".format(msg))
    if not p.exists(SFMT_FILENAME):
        r.urlretrieve(SFMT_URL, SFMT_FILENAME)
    print("[DONE]  {}".format(msg))

    if p.exists(DEST_DIR):
        tdir = tempfile.mkdtemp(prefix='.stashed', dir='.')
        print("Moving {} out of the way to {}".format(DEST_DIR, tdir))
        shutil.move(DEST_DIR, tdir)

    msg = "Unzipping LIBLINEAR source code from {} to {}".format(LL_FILENAME, DEST_DIR)
    print("[START] {}".format(msg))
    try:
        tdir = tempfile.mkdtemp()
        with zipfile.ZipFile(LL_FILENAME) as zfile:
            zip_root = p.commonprefix(zfile.namelist())
            zfile.extractall(path=tdir)
            shutil.move(p.join(tdir, zip_root), DEST_DIR)
    finally:
        shutil.rmtree(tdir)
    print("[DONE]  {}".format(msg))

    msg = "Unzipping SFMT source code from {} to {}".format(SFMT_FILENAME, SFMT_DIR)
    print("[START] {}".format(msg))
    with zipfile.ZipFile(SFMT_FILENAME) as zfile:
        zip_root = p.commonprefix(zfile.namelist())
        zfile.extractall(path=tdir)
        shutil.move(p.join(tdir, zip_root), SFMT_DIR)
    print("[DONE]  {}".format(msg))

    msg = "Applying the PRNG patch to make CV reproducible."
    print("[START] {}".format(msg))
    subprocess.check_call(['patch', '-ts', '-i', p.abspath(PRNG_PATCH),
                           '-d', DEST_DIR])
    print("[DONE]  {}".format(msg))


    msg = "Applying the OpenMP patch for parallelizing CV."
    print("[START] {}".format(msg))
    subprocess.check_call(['patch', '-ts', '-i', p.abspath(OPENMP_PATCH),
                           '-d', DEST_DIR])
    print("[DONE]  {}".format(msg))


    msg = "Building SFMT.o in {} using make".format(SFMT_DIR)
    print("[START] {}".format(msg))
    subprocess.check_call(['make', 'SFMT.o'], cwd=SFMT_DIR)
    print("[DONE]  {}".format(msg))

    msg = "Building the patched LIBLINEAR in {}.".format(DEST_DIR)
    print("[START] {}".format(msg))
    try:
        subprocess.check_call(['make'], cwd=DEST_DIR)
    except:
        if platform.system() == 'Darwin':
            atexit.register(lambda : print("""
If the compilation failed due to lack of OpenMP support in the default C++
compiler, then you may need to use a compiler that supports OpenMP.
One possibility is to install LLVM/Clang with OpenMP support from Homebrew
as follows:

$ brew install libomp llvm

Subsequently, you can rerun this script so that it uses the newly-installed
compilers and the OpenMP binaries using the following command:

$ CXX=/usr/local/opt/llvm/bin/clang++ CC=/usr/local/opt/llvm/bin/clang python installer.py

Please see README.TXT for more information.

"""))
        raise
    print("[DONE]  {}".format(msg))

    if not p.exists('rcv1_train.binary'):
        msg = "Decompressing rcv1_train.binary.bz2"
        print("[START] {}".format(msg))
        with bz2.BZ2File(RCV1_FILENAME, 'rb') as in_file, open('rcv1_train.binary', 'wb') as out_file:
            out_file.write(in_file.read())
        print("[DONE]  {}".format(msg))

    train = p.join(DEST_DIR, 'train')

    msg = "Running the demo for parallelized 5-fold CV."
    print("[START] {}".format(msg))
    for num_threads in [2, 3, 4, 5]:
        print("Number of threads:", num_threads)
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(num_threads)
        subprocess.check_call([train, '-c', '4', '-e', '0.1',
                               '-v', '5', 'rcv1_train.binary'],
                               env=env)
    print("[DONE]  {}".format(msg))
