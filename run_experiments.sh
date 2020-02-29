#!/bin/bash

# Script to reproduce results

for ((i=0;i<5;i+=1))
do 
	python3 run.py \
	--alg "PPO" \
  --env "HalfCheetahBulletEnv-v0" \
  --seed $i \


	python3 run.py \
	--alg "PPO" \
  --env "HopperBulletEnv-v0" \
  --seed $i \

	python3 run.py \
	--alg "PPO" \
  --env "FixedReacherBulletEnv-v0" \
  --seed $i \

done
