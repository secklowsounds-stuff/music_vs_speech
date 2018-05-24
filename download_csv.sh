#!/bin/bash

echo -n "Username: " 
read USER
#stty -echo
#echo -n "password: "
#read PASS
#stty echo

echo "Connecting..."
mkdir -p datasets/SecklowSounds
scp -P 54211 $USER@data.mksmart.org:/var/www/html/secklows-tagging/docs/*.csv datasets/SecklowSounds/docs
echo "Download completed"