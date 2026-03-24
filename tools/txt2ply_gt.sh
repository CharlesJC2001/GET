#!/bin/bash
gt_instance_dir="checkpoints/visualization/gt_instance"
output_dir="checkpoints/visualization/ply/gt_instance"

for txt_file in "$gt_instance_dir"/*.txt; do
	filename=$(basename "$txt_file")
	room_name="${filename%.txt}"
	output_file="$output_dir/$room_name.ply"
	echo "input file:$txt_file"
	echo "room_name:$room_name"
	echo "output_file:$output_file"

	python tools/visualize.py --room_name "$room_name" --out "$output_file"

	if [-f "$output_file"];then
		echo "success: $room_name"
	else
		echo "error:fail to process $room_name"
	fi
	echo "------------------"
done
