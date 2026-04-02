#!/bin/bash

# config
KEY="~/.ssh/karen-hgr-key.pem"
EC2_USER="ec2-user"
EC2_IP="YOUR_INSTANCE_IP"
INSTANCE_ID="i-0abc123def456"
SCRIPT="test.py"
RESULTS_FILE="/home/ec2-user/results.json"
LOCAL_DEST="/Users/karenxu/Documents/School/Spring2026/DS/code/X-RAG/src/results"

echo "Running script on EC2..."
ssh -i $KEY -o StrictHostKeyChecking=no $EC2_USER@$EC2_IP "python ~/your_script.py"

if [ $? -ne 0 ]; then
  echo "Script failed, stopping instance."
  aws ec2 stop-instances --instance-ids $INSTANCE_ID --region ca-central-1
  exit 1
fi

echo "SCPing results file to local..."
scp -i $KEY $EC2_USER@$EC2_IP:$RESULTS_FILE $LOCAL_DEST

if [ $? -ne 0 ]; then
  echo "SCP failed, stopping instance."
  aws ec2 stop-instances --instance-ids $INSTANCE_ID --region ca-central-1
  exit 1
fi
echo "Results saved to $LOCAL_DEST"

echo "Terminating instance..."
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region ca-central-1
echo "Instance terminated."
