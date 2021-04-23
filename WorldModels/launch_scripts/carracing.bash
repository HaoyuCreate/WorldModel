CONFIG_PATH=configs/carracing.config
for i in `seq 1 4`;
do
  echo worker $i
  CUDA_VISIBLE_DEVICES=-1 xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 extract.py -c $CONFIG_PATH &
  sleep 1.0
done
wait
CUDA_VISIBLE_DEVICES=0 python3 vae_train.py -c $CONFIG_PATH
CUDA_VISIBLE_DEVICES=0 python3 series.py -c $CONFIG_PATH
CUDA_VISIBLE_DEVICES=0 python3 rnn_train.py -c $CONFIG_PATH
CUDA_VISIBLE_DEVICES=-1 xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python3 train.py -c $CONFIG_PATH
