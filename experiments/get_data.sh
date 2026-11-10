#!/usr/bin/env bash
# Downloads the two public location files into data/.
set -e
cd "$(dirname "$0")/../data"
B=https://raw.githubusercontent.com/plotly/datasets/master
curl -L -o uber-rides-data1.csv $B/uber-rides-data1.csv
curl -L -o earthquakes-23k.csv $B/earthquakes-23k.csv
