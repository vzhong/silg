#!/bin/bash

mkdir -p data
mkdir -p graph

wget -O data/dev.json https://github.com/lil-lab/touchdown/blob/2ad8ef2664d1aa8ff1a8eebf88b9e88b2cb1d1c6/data/dev.json?raw=true
wget -O data/test.json https://github.com/lil-lab/touchdown/blob/2ad8ef2664d1aa8ff1a8eebf88b9e88b2cb1d1c6/data/test.json?raw=true
wget -O data/train.json https://github.com/lil-lab/touchdown/blob/2ad8ef2664d1aa8ff1a8eebf88b9e88b2cb1d1c6/data/train.json?raw=true

wget -O graph/links.txt https://github.com/lil-lab/touchdown/blob/2ad8ef2664d1aa8ff1a8eebf88b9e88b2cb1d1c6/graph/links.txt?raw=true
wget -O graph/nodes.txt https://github.com/lil-lab/touchdown/blob/2ad8ef2664d1aa8ff1a8eebf88b9e88b2cb1d1c6/graph/nodes.txt?raw=true