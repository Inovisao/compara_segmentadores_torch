library("ggplot2")
library("gridExtra")
library("Metrics")

#
#   BOXPLOT DO DESEMPENHO ENTRE MÉTRICAS
#
dados <- read.table('../results_dl/results.csv',sep=',',header=TRUE)

metricas <- c("precision","recall","fscore","miou")
classes <- unique(dados$classe)


for (classe in classes) {
    graficos <- list()
    i <- 1
    for (metrica in metricas) {
        
        print(metrica)

        dados_classe <- dados[dados$classe == classe, ]

        if (any(!is.na(dados_classe[[metrica]]))) {    
            TITULO = sprintf("Boxplot for %s",metrica)
            g <- ggplot(dados_classe, aes_string(x="architecture", y = metrica, fill="optimizer"))+
            geom_boxplot()+
            scale_fill_brewer(palette="Purples")+
            labs(title=TITULO, x="Models", y = metrica)+
            theme(legend.position="right")+
            theme(plot.title = element_text(hjust = 0.5))

            graficos[[i]] <- g
            i = i + 1
            }
    }

    if (length(graficos) > 0) { 
        g <- grid.arrange(grobs=graficos, ncol = 2)
        ggsave(paste(sprintf("../results_dl/boxplot_%s.png",classe), sep=""),g, width = 12, height = 10)
        print(g)
    }
    
}

#
#
#