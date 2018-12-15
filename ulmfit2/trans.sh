#!/usr/bin/env bash
# Script to download a Wikipedia dump

TRANS="py-googletrans"
# Check if directory exists
#if [ ! -d "${TRANS}" ]; then
#  git clone https://github.com/ssut/py-googletrans.git
#  cd "${TRANS}"
#  python setup.py install
#  cd ..
#fi
cd "${TRANS}"
python setup.py install
cd ..
