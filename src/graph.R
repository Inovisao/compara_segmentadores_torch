library("ggplot2")
library("gridExtra")
library("Metrics")

#
#   BOXPLOT DO DESEMPENHO ENTRE MÉTRICAS
#
dados <- read.table('../results_dl/results.csv',sep=',',header=TRUE)

metricas <- list("precision","recall","fscore","miou")
graficos <- list()
i <- 1

for (metrica in metricas) {
    
    print(metrica)
    TITULO = sprintf("Boxplot for %s",metrica)
    g <- ggplot(dados, aes_string(x="architecture", y = metrica, fill="optimizer"))+
    geom_boxplot()+
    scale_fill_brewer(palette="Purples")+
    labs(title=TITULO, x="Models", y = metrica)+
    theme(legend.position="right")+
    theme(plot.title = element_text(hjust = 0.5))

    graficos[[i]] <- g 
    i = i + 1
}

g <- grid.arrange(grobs=graficos, ncol = 2)
ggsave(paste("../results_dl/boxplot.png", sep=""),g, width = 12, height = 10)
print(g)

#
#
#