##  Código baseado no tutorial do Sovit Ranjan Rath com algumas alterações (https://debuggercafe.com/multi-class-semantic-segmentation-training-using-pytorch#download-code)

## Organizando os dados e rodando experimento:
    - Coloque as imagens separadas em uma pasta por classe dentro de ./data/all. Certifique-se de que os nomes dos arquivos não contêm espaços.
    - Rode o script split.sh, passando o número de dobras como argumento -k (ex.: ./split.sh -k 10). Frequentemente se utilizam dez dobras.
    - Selecione as redes (arquiteturas e otimizadores) a serem testadas em roda.sh.
    - Altere os hiperparâmetros em hyperparameters.py.
    - Se necessário, altere os hiperparâmetros dos otimizadores diretamente em optimizers.py.
    - Rode o script rodaCruzada.sh, passando o número de dobras como argumento -k (ex.: ./rodaCruzada.sh -k 10).


## Preparando o ambiente:
    - Abra o terminal e rode essa linha: conda create -n segmentacao
    - Logo após, rode essa linha: conda activate segmentacao
    - Rode o script env.sh dentro de utils para preparar o ambiente com as bibliotecas


## Adicionando mais arquiteturas.

- Para adicionar uma nova arquitetura, defina uma função em architectures.py. Instancie a arquitetura e programe a alteração da primeira e da última camadas. Registre a nova arquitetura no dicionário que consta em arch_optim.py.
- Para adicionar um novo otimizador, defina uma função em optimizers.py. Os hiperparâmetros do otimizador devem estar declarados explicitamente, mesmo que o valor atribuído seja o valor padrão.

## Informações adicionais.

- Os resultados relativos à dobra em execução são colocados na pasta ./results. Os resultados por dobra são colocados na pasta ./resultsNfolds após a execução completa da dobra.

## Troubleshooting.

- Certifique-se de que o ambiente correto está ativo.
- Algumas arquiteturas exigem configurações específicas, especialmente no que tange ao tamanho de imagem. Assim, por exemplo, a arquitetura coat_tiny exige imagens com dimensões exatamente iguais a (224, 224). Além disso, várias arquiteturas exigem que as imagens tenham um tamanho mínimo específico.
- Caso a memória da GPU seja insuficiente, diminua o tamanho do lote nos hiperparâmetros (batch size).
