#!/bin/bash
set -e
source /global/common/software/lsst/cori-haswell-gcc/stack/setup_current_sims.sh "" &> /dev/null
setup lsst_distrib &> /dev/null
cd "$(dirname "$0")"
python -W ignore -c "from make_cutout import main; main()" "$@"
