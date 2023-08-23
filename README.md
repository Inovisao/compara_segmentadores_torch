##  Código baseado no tutorial do Sovit Ranjan Rath com algumas alterações (https://debuggercafe.com/multi-class-semantic-segmentation-training-using-pytorch#download-code)

##  Novas implementações:
    Redes novas
    Script para separar as imagens e as labels
    Script para rodar o código
    ...
    (IDEIAS FUTURAS -> NOVOS OTIMIZADORES, NOVAS MÉTRICAS)

## Organizando os dados:
    - Coloque as imagens dentro de input/imagens e as labels em input/labels (ambos em formato png)
    - Rode o script split.sh presente na pasta utils para separar as imagens de treino e validação

## Instalação:
    - Abra o terminal e rode essa linha: conda create -n segmentacao
    - Logo após, rode essa linha: conda activate segmentacao
    - Rode o script env.sh dentro de utils para preparar o ambiente com as bibliotecas