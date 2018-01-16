#!/bin/bash

if [ "$TRAVIS_BRANCH" == "master" ]; then
  az login -u $AZURE_LOGIN_USR -p $AZURE_LOGIN_PWD
  az vm start -g $AZURE_VM_GROUP -n $AZURE_VM_NAME
  sleep 60
  sshpass -p $AZURE_SSH_PWD ssh $AZURE_SSH_USR@$AZURE_HOST -o stricthostkeychecking=no 'bash deblurring/run.sh'
  az vm stop -g $AZURE_VM_GROUP -n $AZURE_VM_NAME
  az vm deallocate -g $AZURE_VM_GROUP -n $AZURE_VM_NAME
fi