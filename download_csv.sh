#!/bin/bash

echo -n "Username: " 
read USER
#stty -echo
#echo -n "password: "
#read PASS
#stty echo

echo "Connecting..."
mkdir -p data/SecklowSounds
scp -P 54211 $USER@data.mksmart.org:/var/www/html/secklows-tagging/docs/*.csv data/SecklowSounds/docs
echo "Download completed"