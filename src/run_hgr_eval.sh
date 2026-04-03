#!/bin/bash

# config
KEY="~/.ssh/karen-hgr-key.pem"
EC2_USER="ec2-user"
INSTANCE_ID="i-0f07890683eae8a66"
SCRIPT="test_bash_script.py"
RESULTS_FILE="/home/ec2-user/X-RAG/src/results/test_bash.json"
LOCAL_DEST="/Users/karenxu/Documents/School/Spring2026/DS/code/X-RAG/src/results"

echo "Starting instance $INSTANCE_ID"
aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region ca-central-1

echo "Fetching IP"
EC2_IP=""
i=0
while :
do 
  DESC=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region ca-central-1)
  # echo "desc is $DESC"

  STATE=$(echo "$DESC" | jq -r ".Reservations[0].Instances[0].State.Name")
  echo "State is $STATE"
    
  if [ "$STATE" == "running" ]; then
    EC2_IP=$(echo "$DESC" | jq -r ".Reservations[0].Instances[0].PublicIpAddress")
    echo "IP is $EC2_IP"
    break
  else
    echo "Retrying... ($i)"
    i=$((i+1))
    sleep 2
  fi
done

j=0
until ssh -i $KEY -o StrictHostKeyChecking=no $EC2_USER@$EC2_IP \
  "cd ~/X-RAG/src && uv run python3 $SCRIPT"
do
  echo "Try ssh $EC2_USER@$EC2_IP ($j)"
  j=$((j+1))
  sleep 5
done

echo "Running script on EC2..."
ssh -i $KEY -o StrictHostKeyChecking=no $EC2_USER@$EC2_IP \
  "cd ~/X-RAG/src && nohup uv run python3 $SCRIPT OURS_TEST" &

wait

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
aws ec2 stop-instances --instance-ids $INSTANCE_ID --region ca-central-1
echo "Success, instance stopped."
