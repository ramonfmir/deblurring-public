#!/bin/bash

if [ "$TRAVIS_BRANCH" == "master" ]; then
  az login -u rf1515@ic.ac.uk -p cpOO9506
  az vm start -g frobenius -n tobal
  sleep 60
  sshpass -p password12345678! ssh funny@52.170.199.130 -o stricthostkeychecking=no 'bash deblurring/run.sh'
  az vm stop -g frobenius -n tobal
  az vm deallocate -g frobenius -n tobal
fi
