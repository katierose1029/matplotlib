#! /bin/bash

set -ev

# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

# The number of processes is hardcoded, because using too many causes the
# Travis VM to run out of memory (since so many copies of inkscape and
# ghostscript are running at the same time).

if [[ $BUILD_DOCS == false ]]; then
  if [[ $DELETE_FONT_CACHE == 1 ]]; then
    rm -rf ~/.cache/matplotlib
  fi
  # Workaround for pytest-xdist flaky collection order
  # https://github.com/pytest-dev/pytest/issues/920
  # https://github.com/pytest-dev/pytest/issues/1075
  export PYTHONHASHSEED=$(python -c 'import random; print(random.randint(1, 4294967295))')
  echo PYTHONHASHSEED=$PYTHONHASHSEED

  echo The following args are passed to pytest $PYTEST_ARGS $RUN_PEP8
  if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    pytest $PYTEST_ARGS $RUN_PEP8
  else
    gdb -return-child-result -batch -ex r -ex bt --args python $PYTHON_ARGS -m pytest $PYTEST_ARGS $RUN_PEP8
  fi
else
  cd doc
  python make.py html
  # We don't build the LaTeX docs here, so linkchecker will complain
  touch build/html/Matplotlib.pdf
  # Linkchecker only works with python 2.7 for the time being
  source ~/virtualenv/python2.7/bin/activate
  pip install pip --upgrade
  # linkchecker is currently broken with requests 2.10.0 so force an earlier version
  pip install $PRE requests==2.9.2 linkchecker
  linkchecker build/html/index.html
fi
