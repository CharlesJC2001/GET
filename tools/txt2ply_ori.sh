#!/bin/bash
pred_instance_dir="checkpoints/visualization/pred_instance"
output_dir="checkpoints/visualization/ply/origin_pc"

for txt_file in "$pred_instance_dir"/*.txt; do
	filename=$(basename "$txt_file")
	room_name="${filename%.txt}"
	output_file="$output_dir/$room_name.ply"
	echo "input file:$txt_file"
	echo "room_name:$room_name"
	echo "output_file:$output_file"

	python tools/visualize.py --room_name "$room_name" --task "origin_pc" --out "$output_file"

	if [-f "$output_file"];then
		echo "success: $room_name"
	else
		echo "error:fail to process $room_name"
	fi
	echo "------------------"
done
