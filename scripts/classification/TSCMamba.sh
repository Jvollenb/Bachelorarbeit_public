if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/classification" ]; then
    mkdir ./logs/classification
fi
if [ ! -d "./csv_results" ]; then
    mkdir ./csv_results
fi

if [ ! -d "./csv_results/classification" ]; then
    mkdir ./csv_results/classification
fi

model_name=TSCMamba

if [ -z "$1" ]; then
    echo "Fehler: Kein Datensatzname übergeben."
    echo "Verwendung: $0 <dataset_name>"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Fehler: Kein train_eps Wert übergeben."
    echo "Verwendung: $0 <dataset_name> <train_eps>"
    exit 1
fi

if [ -z $3 ]; then
    echo "Fehler: Kein use_noise Wert übergeben."
    echo "Verwendung: $0 <dataset_name> <train_eps> <use_noise>"
    exit 1
fi

if [ -z "$4" ]; then
    echo "Fehler: Kein weight_decay Wert übergeben."
    echo "Verwendung: $0 <dataset_name> <train_eps> <use_noise> <weight_decay>"
    exit 1
fi

if [ -z "$5" ]; then
    echo "Fehler: Kein dropout Wert übergeben."
    echo "Verwendung: $0 <dataset_name> <train_eps> <use_noise> <weight_decay> <dropout>"
    exit 1
fi

if [ -z "$6" ]; then
    echo "Fehler: Kein learning_rate Wert übergeben."
    echo "Verwendung: $0 <dataset_name> <train_eps> <use_noise> <weight_decay> <dropout> <learning_rate>"
    exit 1
fi

dataset_name=$1
root_path_name=./datasets/$dataset_name
model_id_name=$dataset_name
data_name=UEA

random_seed=2025

train_eps=$2
use_noise=$3
weight_decay=$4
dropout=$5
learning_rate=$6



python3 -u run.py \
    --task_name classification \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --model_id $model_id_name \
    --model $model_name \
    --data $data_name \
    --dropout $dropout \
    --dconv 4 \
    --d_state 32 \
    --e_fact 2 \
    --projected_space 64 \
    --num_mambas 1 \
    --des 'Exp' \
    --lradj 'cosine' \
    --comment 'Max pooling after Mambas' \
    --no_rocket 0 \
    --additive_fusion 1 \
    --max_pooling 1 \
    --white_noise $use_noise \
    --train_epochs 1000 \
    --train_eps $train_eps \
    --weight_decay $weight_decay \
    --delta 0.0001 \
    --patience 50 \
    --itr 10 --batch_size 32 --learning_rate $learning_rate >logs/classification/$model_name'_'$model_id_name.log
