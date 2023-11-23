##  Código baseado no tutorial do Sovit Ranjan Rath com algumas alterações (https://debuggercafe.com/multi-class-semantic-segmentation-training-using-pytorch#download-code)

##  Novas implementações:
    Redes novas
    Script para separar as imagens por dobras
    Script para rodar o código


## Organizando os dados:
    - Coloque as imagens dentro de data/all/imagens e as labels em data/labels (ambos em formato png)
    - Rode o script split.sh presente na pasta utils para separar as imagens por dobras. Lembre-se de passar número de dobras 
    como parâmetro. ex.: split.sh -k 10  ### Criando 10 dobras.

## Preparando o ambiente:
    - Abra o terminal e rode essa linha: conda create -n segmentacao
    - Logo após, rode essa linha: conda activate segmentacao
    - Rode o script env.sh dentro de utils para preparar o ambiente com as bibliotecas

## Rodando um experimento:
    - Ajuste os hiperparâmetros em src/data_hyperparameters.py
    - Após organizar os dados e preparar o ambiente, abra o arquivo roda.sh na pasta RUN e escolha as arquiteturas e otimizadores
    - Abra o terminal e execute o rodaCruzada.sh passando como argumento o número de dobras. (Exemplo: ./rodaCruzada.sh -k 5)
