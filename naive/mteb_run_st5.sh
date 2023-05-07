nohup python -u st5_temp.py \
    --model_name_or_path t5-base \
    --task_types STS \
    --gpu 1 \
    --output_folder ./STS_st5 \
    >& STS_st5.out & 
    