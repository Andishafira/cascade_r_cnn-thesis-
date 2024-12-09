NUM_GPUS=$1
export PORT=9487
export MASTER_PORT=9487

set -e
start=`date +%s`
>&2 echo "---------------------" 
>&2 echo "cascade_original.json" 
>&2 echo "---------------------" 
echo "---------------------" 
echo "cascade_original.json" 
echo "---------------------" 

# cascade_original.json
bash /content/cascade_r_cnn-thesis-/tools/dist_test.sh /content/cascade_r_cnn-thesis-/configs/mva2023/cascade_rcnn_r50_fpn_40e_coco_nwd_finetune.py $NUM_GPUS --format-only --eval-options jsonfile_prefix=cascade_original