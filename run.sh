#!/bin/sh

cd ./front-end/
python front.py
cd ../middle-end/
python ./middle.py
'''
cd ../back-end/
python ./back.py
rm -fr ../front-end/frontOut.txt
rm -fr ../middle-end/middleOut.txt
rm -fr ../back-end/backOut.txt
'''
