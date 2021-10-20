#!/usr/bin/env bash
set -xev

ROOT=$PWD

mkdir -p $ROOT/ext

# RTFM
echo installing RTFM
DWORK=$ROOT/ext/rtfm
REPO=https://github.com/facebookresearch/RTFM
if [ ! -d "$DWORK" ]
then
  git clone $REPO $DWORK
  cd $DWORK && git checkout 58f17955595b5a127c96d045d896fcbcc7d4b570
  pip install -e $DWORK
  cd $ROOT
fi

# Messenger
echo installing Messenger
DWORK=$ROOT/ext/messenger-emma
REPO=https://github.com/ahjwang/messenger-emma
if [ ! -d "$DWORK" ]
then
  git clone $REPO $DWORK
  cd $DWORK && git checkout ba411bfb8d71146ba41442c47441ec2f2873f806
  pip install -e $DWORK
fi

# Nethack
echo installing Nethack
DWORK=$ROOT/ext/nethack
REPO=https://github.com/facebookresearch/nle
if [ ! -d "$DWORK" ]
then
  git clone $REPO $DWORK --recursive
  cd $DWORK
  git checkout cfa10594d4ad85b18c8151f4942790bfdd05f3bf
  pip install -e .[dev]
  cd $ROOT
fi

# ALFWorld
echo installing ALFWorld
DWORK=$ROOT/ext/alfworld
REPO=https://github.com/alfworld/alfworld
if [ ! -d "$DWORK" ]
then
  pip install https://github.com/MarcCote/downward/archive/faster_replan.zip
  pip install https://github.com/MarcCote/TextWorld/archive/handcoded_expert_integration.zip
  git clone $REPO $DWORK
  cd $DWORK
  git checkout 2ddb514531cd520061dd77b845426ee41bd93466
  # we're gonna remove dependencies on specific versions of pytorch etc
  cp requirements.txt orig_requirements.txt
  grep -v "numpy" requirements.txt > tmp ; mv tmp requirements.txt
  grep -v "gym" requirements.txt > tmp ; mv tmp requirements.txt
  grep -v "torch" requirements.txt > tmp ; mv tmp requirements.txt
  grep -v "tensorboard" requirements.txt > tmp ; mv tmp requirements.txt
  pip install -e .
  cd $ROOT
fi

# Touchdown
if [ ! -e silg/envs/touchdown/data/train.json ]
then
  cd silg/envs/touchdown && bash download.sh
fi
