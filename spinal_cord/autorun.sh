#!/bin/bash
export MPLBACKEND="agg"
for i in 1200; do
	python3 run.py $i
done
