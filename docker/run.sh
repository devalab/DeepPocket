#!/bin/sh

docker build -t pockets .

cd PocketPredict

python dock.py