
STORE_PARAM_NAME='bestParam_ckpt.pth'
BOX_TYPE='text' #choose from text and normal
BOX_TYPE_SHOW='box_type'
DATA_DIR='/workspace/czy/code/TextDetection/data/'


LOG_STORE_PATH='./LOG/'
if [ ! -d $LOG_STORE_PATH  ];then
  mkdir $LOG_STORE_PATH
fi
PARAM_STORE_PATH='./checkpoint/'
if [ ! -d $PARAM_STORE_PATH  ];then
  mkdir $PARAM_STORE_PATH
fi
STORE_PARAM_NAME=$PARAM_STORE_PATH$BOX_TYPE_SHOW-$BOX_TYPE-$STORE_PARAM_NAME


BATCH_SIZE=64
NUM_EPOCHS=800
NUM_WORKERS=4
GPU_ID=1

LOG_NAME=$LOG_STORE_PATH$BATCH_SIZE-$NUM_EPOCHS-boxtype-$BOX_TYPE.log

python -m train \
--num_epochs $NUM_EPOCHS \
--batchsize $BATCH_SIZE \
--num_workers $NUM_WORKERS \
--storeParamName $STORE_PARAM_NAME \
--gpu_id $GPU_ID \
--lr 1e-3 \
--box_type $BOX_TYPE \
--data_dir $DATA_DIR \
| tee -a $LOG_NAME




