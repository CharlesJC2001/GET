#!/bin/bash
pred_instance_dir="test_GET/pred_instance"
output_dir="ply/test_instance"

for txt_file in "$pred_instance_dir"/*.txt; do
	filename=$(basename "$txt_file")
	room_name="${filename%.txt}"
	output_file="$output_dir/$room_name.ply"
	echo "input file:$txt_file"
	echo "room_name:$room_name"
	echo "output_file:$output_file"

	python tools/visualize.py --prediction_path "test_GET" --room_name "$room_name" --task "instance_pred" --out "$output_file"

	if [-f "$output_file"];then
		echo "success: $room_name"
	else
		echo "error:fail to process $room_name"
	fi
	echo "------------------"
done
