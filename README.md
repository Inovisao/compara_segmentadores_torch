## Compara_segmentadores_torch
    - Código baseado no tutorial do Sovit Ranjan Rath(https://debuggercafe.com/multi-class-semantic-segmentation-training-using-pytorch#download-code) com algumas alterações feitas para atender a demanda do grupo.

## Preparando o ambiente:
    - Primeiro instale todas as dependências necessárias para o funcionamento do código, para isso leia o arquivo install.txt

## Rodando
    - Após obter as imagens e o arquivo COCO JSON (Roboflow), você deve salvar esses arquivos em data/all/imagens (o dataset baixado do roboflow vem com uma pasta train por padrão, tem que renomear para imagens).Certifique-se de que os nomes dos arquivos não contêm espaços.
    - Rode o script split.sh que se encontra dentro do diretório utils/, passando o número de dobras como argumento -k (ex.: ./splitFolds.sh -k 10). Por padrão 3 dobras são geradas pelo script, adapte conforme a necessidade.
    - Escolha as arquiteturas, otimizadores e a taxa de aprendizagem em roda.sh
    - Defina os hiperparâmetros modificando o arquivo python em src/data_hyperparameters.py
    - Rode o script rodaCruzada que se encontra dentro do diretório run/, você deve utilizar a mesma quantidade de dobras que foi definida para o split(ex.: bash rodaCruzada.sh -k 10), por padrão está definido com 3 dobras, ajuste conforme a necessidade.
    - Para gerar os gráficos e ANOVA referente ao experimento, rode o script graph.R dentro do diretório scr/, através do comando: rscript graph.R

## Informações adicionais.

- Os resultados principais(csv,previsões,máscaras) bem como os gráficos gerados pelo R são colocados na pasta ./results_dl. Os resultados por dobra são colocados na pasta ./resultsNfolds após a execução completa da dobra.

## Troubleshooting.

- Certifique-se de que o ambiente correto está ativo.
- Algumas arquiteturas exigem configurações específicas, especialmente no que tange ao tamanho de imagem. Assim, por exemplo, a arquitetura coat_tiny exige imagens com dimensões exatamente iguais a (224, 224). Além disso, várias arquiteturas exigem que as imagens tenham um tamanho mínimo específico.
- Caso a memória da GPU seja insuficiente, diminua o tamanho do lote nos hiperparâmetros (batch size).
