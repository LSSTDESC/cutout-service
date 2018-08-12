#!/bin/sh
chgrp -R lsst .
chmod g+rs .
chmod g+r *
chmod g+x run.sh 
chmod o+rx .
chmod o+r index.html
