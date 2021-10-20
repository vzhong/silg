#!/usr/bin/env bash
set -vex

my_link=$PWD/cache

if [ -L ${my_link} ] ; then
  if [ -e ${my_link} ] ; then
    echo "your $my_link is set!"
  else
    echo "your $my_link is broken!"
  fi
elif [ -e ${my_link} ] ; then
  echo "your $my_link is not a symlink!"
else
  echo "linking $my_link to $PWD/mycache. You might want to move the real folder to /tmp if you are on NFS!"
  ln -s $PWD/mycache $my_link
fi

echo downloading alfworld data
if [ ! -d "$my_link/alfworld/detectors" ] ; then
  ALFWORLD_DATA=$my_link/alfworld alfworld-download
fi

echo downloading touchdown
if [ ! -e "silg/envs/touchdown/data/test.json" ] ; then
  cd silg/envs/touchdown && bash download.sh
fi

mkdir -p $my_link/touchdown

if [ ! -e "$my_link/touchdown/pca_10.npz" ] ; then
  wget https://www.dropbox.com/s/7gtmny90m4bto5e/pca_10.npz?dl=0 -O $my_link/touchdown/pca_10.npz  # comment this out if you do not plan on using VisTouchdown
fi
if [ ! -e "$my_link/touchdown/shortest_paths.npz" ] ; then
  wget https://www.dropbox.com/s/pxih298mj2pwgrm/shortest_paths.npz?dl=0 -O $my_link/touchdown/shortest_paths.npz
fi
if [ ! -e "$my_link/touchdown/maj_ds_a10.npz" ] ; then
  wget https://www.dropbox.com/s/pe8iogx4tg2s2tv/maj_ds_a10.npz?dl=0 -O $my_link/touchdown/maj_ds_a10.npz
fi

mkdir -p $my_link/tokenizer
if [ ! -e "$my_link/tokenizer.json" ] ; then
  wget https://huggingface.co/bert-base-uncased/raw/main/vocab.txt -O $my_link/tokenizer/vocab.txt
  wget https://huggingface.co/bert-base-uncased/raw/main/tokenizer_config.json -O $my_link/tokenizer/config.json
  wget https://huggingface.co/bert-base-uncased/raw/main/tokenizer.json -O $my_link/tokenizer/tokenizer.json
fi
