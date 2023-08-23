#!/bin/bash
#para rodar esse script as imagens devem estar em /input/imagens e as labels em /input/labels
SRC_IMG_DIR="../input/imagens"
DST_DIR="../input/split_data"
SPLITS=("train" "val")
TRAIN_PERCENT=80

# Criar diretórios de destino
mkdir -p "$DST_DIR/images"
mkdir -p "$DST_DIR/labels"

# Ler todas as imagens no diretório de origem
all_images=($(ls "$SRC_IMG_DIR"))

# Calcular quantas imagens para treinamento e validação
num_images=${#all_images[@]}
num_train=$((num_images * TRAIN_PERCENT / 100))
num_val=$((num_images - num_train))

# Embaralhar a lista de imagens
shuf_images=($(shuf -e "${all_images[@]}"))

# Separar as imagens em conjuntos de treinamento e validação
train_images=("${shuf_images[@]:0:num_train}")
val_images=("${shuf_images[@]:num_train:num_val}")

# Função para copiar dados para os diretórios de destino
function copy_data {
    for data in "${@:2}"; do
        data_name="${data%.*}"
        cp "$SRC_IMG_DIR/$data_name.png" "$DST_DIR/images/$1/$data_name.png"
        cp "../input/labels/$data_name.png" "$DST_DIR/labels/$1/$data_name.png"
    done
}

for split in "${SPLITS[@]}"; do
    mkdir -p "$DST_DIR/images/$split"
    mkdir -p "$DST_DIR/labels/$split"

    if [ "$split" == "train" ]; then
        copy_data "train" "${train_images[@]}"
    else
        copy_data "val" "${val_images[@]}"
    fi
done
