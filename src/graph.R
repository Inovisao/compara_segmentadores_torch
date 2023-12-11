library("ggplot2")
library("gridExtra")
library("Metrics")
library("ExpDes.pt")
library("kableExtra")
library(data.table)
#
#   BOXPLOT DO DESEMPENHO ENTRE MÉTRICAS
#
dados <- read.table('../results_dl/results.csv',sep=',',header=TRUE)

desconsiderar_na_anova <- c("média", "fundo")


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
possible_factors <- list("learning_rate","architecture","optimizer")
factors <- list()
i <- 1
for (possible_factor in possible_factors){
    if(length(unique(dados[,possible_factor])) > 1){
        factors[i] <- possible_factor
        i <- i + 1
    }

}


all_clases <- unique(dados[, "classe"])
classes_anova <- list()
for (c in all_clases) {
    if (!c %in% desconsiderar_na_anova) {
      classes_anova <- append(classes_anova, c)
    }
}

one_way_anova <- function(dataframe, factor) {
  sink("../results_dl/one_way.txt")
  
  cat(sprintf('\n\n====>>> TESTING: PRECISION =============== \n\n'))
  fprecision <- aov(as.formula(paste("precision~", factor)), data=dataframe)
  print(summary(fprecision))

  cat(sprintf('\n\n====>>> TESTING: RECALL ================ \n\n'))
  frecall <- aov(as.formula(paste("recall~", factor)), data=dataframe)
  print(summary(frecall))

  cat(sprintf('\n\n====>>> TESTING: F-SCORE =============== \n\n'))
  ffscore <- aov(as.formula(paste("fscore~", factor)), data=dataframe)
  print(summary(ffscore))
  
  sink()
}

two_way_anova <- function(dataframe, factors) {
  # Applies two way anova to any two factors given in a list.
  # The response variables are precision, recall and fscore.
  sink("../results_dl/two_way.txt")    
  
  cat(sprintf('\n\n====>>> TESTING: PRECISION =============== \n\n'))
  fat2.dic(dataframe[, sprintf("%s", factors[1])], 
           dataframe[, sprintf("%s", factors[2])], 
           dataframe$precision, 
           quali=c(TRUE, TRUE),
           mcomp="sk")
  
  cat(sprintf('\n\n====>>> TESTING: RECALL =============== \n\n'))
  fat2.dic(dataframe[, sprintf("%s", factors[1])], 
           dataframe[, sprintf("%s", factors[2])], 
           dataframe$recall, 
           quali=c(TRUE, TRUE),
           mcomp="sk") 
  
  cat(sprintf('\n\n====>>> TESTING: FSCORE =============== \n\n'))
  fat2.dic(dataframe[, sprintf("%s", factors[1])], 
           dataframe[, sprintf("%s", factors[2])], 
           dataframe$fscore, 
           quali=c(TRUE, TRUE),
           mcomp="sk")
  
  sink()
}

three_way_anova <- function(dataframe, factors) {
  # Applies three way anova to any three factors given in a list.
  # The response variables are precision, recall and fscore. 
  
  sink("../results_dl/three_way.txt")
  
  cat(sprintf('\n\n====>>> TESTING: PRECISION =============== \n\n'))
  fat3.dic(dataframe[, sprintf("%s", factors[1])], 
           dataframe[, sprintf("%s", factors[2])], 
           dataframe[, sprintf("%s", factors[3])], 
           dataframe$precision, 
           quali=c(TRUE, TRUE, TRUE), 
           mcomp="sk") 
  
  cat(sprintf('\n\n====>>> TESTING: RECALL ================= \n\n'))
  fat3.dic(dataframe[, sprintf("%s", factors[1])], 
           dataframe[, sprintf("%s", factors[2])], 
           dataframe[, sprintf("%s", factors[3])], 
           dataframe$recall, 
           quali=c(TRUE, TRUE, TRUE), 
           mcomp="sk") 
  
  cat(sprintf('\n\n====>>> TESTING: FSCORE ================= \n\n'))
  fat3.dic(dataframe[, sprintf("%s", factors[1])], 
           dataframe[, sprintf("%s", factors[2])], 
           dataframe[, sprintf("%s", factors[3])], 
           dataframe$fscore, 
           quali=c(TRUE, TRUE, TRUE), 
           mcomp="sk") 
  
  sink()
}

# Apply anova according to the number of factors.
if (length(factors) == 1){
  one_way_anova(dados, factors)
} else if (length(factors) == 2) {
  two_way_anova(dados, factors)
} else if (length(factors) == 3) {
  three_way_anova(dados, factors)
} else {
  print("Incorrect number of factors. Anova could not be applied.")
}

###########################################################
# Get some statistics.
###########################################################


options(width=10000) # Change line width

dt <- data.table(dados)

precision_statistics <- dt[, list(median=median(precision), IQR=IQR(precision), mean=mean(precision), sd=sd(precision)), by=.(learning_rate, architecture, optimizer)]

recall_statistics <- dt[, list(median=median(recall), IQR=IQR(recall), mean=mean(recall), sd=sd(recall)), by=.(learning_rate, architecture, optimizer)]

fscore_statistics <- dt[, list(median=median(fscore), IQR=IQR(fscore), mean=mean(fscore), sd=sd(fscore)), by=.(learning_rate, architecture, optimizer)]

# Create a .txt with the statistics.
sink('../results_dl/statistics.txt')

cat("\n[ Statistics for precision ]-----------------------------\n")
print(precision_statistics)

cat("\n[ Statistics for recall]-----------------------------\n")
print(recall_statistics)

cat("\n[ Statistics for fscore]-----------------------------\n")
print(fscore_statistics)
sink()

# Save the statistics in LaTeX table format.
sink("../results_dl/statistics_for_latex.txt")

cat(kbl(precision_statistics, caption="Statistics for precision",
      format="latex",
      col.names=c("Learning rate", "Architecture", "Optimizer", "Median", "IQR", "Mean", "SD"),
      align="r"))

cat(kbl(recall_statistics, caption="Statistics for recall",
      format="latex",
      col.names=c("Learning rate", "Architecture", "Optimizer", "Median", "IQR", "Mean", "SD"),
      align="r"))

cat(kbl(fscore_statistics, caption="Statistics for fscore",
      format="latex",
      col.names=c("Learning rate", "Architecture", "Optimizer", "Median", "IQR", "Mean", "SD"),
      align="r"))

sink()

