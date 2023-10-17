# CLASSIFICADORES QUE VOCÊ QUER TESTAR
#arqs=(deeplabv3_resnet101 deeplabv3_resnet50 fcn_resnet50)
arqs=(deeplabv3_resnet101)

mkdir -p ../results
rm -rf ../results/*
mkdir -p ../results/history

# OPTIMIZADORES
#opt=(adam sgd adagrad lion sam)
opt=(adam)

# LEARNING RATES
learning_rates=(0.001)

cd ../src
for lr in "${learning_rates[@]}"
do
    for i in "${arqs[@]}"
    do
        for k in "${opt[@]}"
        do
            echo 'Running' ${lr} ' ' ${i} ' ' ${k} ' see results in folder ../results/'
            python main.py -a $i -o $k -r $1 -l $lr > >(tee -a ../results/${i}_${k}_${lr}.output) 2> >(tee ../results/error_log_${i}_${k}_${lr}.txt >&2)
        done
    done
done

cd ../run

# Utilizar o comando abaixo para ir removendo os checkpoints.
# Caso o experimento seja muito grande, os checkpoints podem chegar a
# dezenas de gigabytes.
# rm ../model_checkpoints/*
