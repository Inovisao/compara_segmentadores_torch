#!/bin/bash

# Define os valores padrão para os argumentos
epochs=5
lr=0.001
batch=4
imgsz=512
scheduler=""

# Função de ajuda
usage() {
    echo "Uso: $0 [--epochs EPOCHS] [--lr LR] [--batch BATCH] [--imgsz IMGSZ] [--scheduler]"
    exit 1
}

# Processa os argumentos da linha de comando
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)
            epochs="$2"
            shift 2
            ;;
        --lr)
            lr="$2"
            shift 2
            ;;
        --batch)
            batch="$2"
            shift 2
            ;;
        --imgsz)
            imgsz="$2"
            shift 2
            ;;
        --scheduler)
            scheduler="true"
            shift
            ;;
        *)
            usage
            ;;
    esac
done

# Inicia o arquivo main.py com os argumentos definidos
python ../src/main.py --epochs "$epochs" --lr "$lr" --batch "$batch" --imgsz "$imgsz" "$scheduler"
