#!/bin/bash
export MPLBACKEND="agg"
for i in `cat autoparams.csv`; do
	python3 run.py 250 $i
done
