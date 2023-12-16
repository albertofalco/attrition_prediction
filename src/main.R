########################################################################################################
########################################################################################################
#
# MAESTRÍA EN MÉTODOS CUANTITATIVOS PARA LA GESTIÓN Y ANÁLISIS DE DATOS EN ORGANIZACIONES
## M72.1 11 MODELOS DE REGRESIÓN GENERALIZADOS
#
# Predicción de la rotación laboral no deseada
## Evaluación de modelos de regresión generalizados de clasificación binaria y de variables exógenas
## para la estimación de la probabilidad de cese no deseado del personal
#
# Autor: Alberto Osvaldo Falco
# Fecha: 21/11/2023

########################################################################################################
########################################################################################################

# Importación de librerias

library(dplyr)
library(ggplot2)
library(gridExtra)
library(caret)
library(DescTools)
library(ROCR)
library(lmtest)
library(glue)
library(extrafont)

###############################################################################

# Configuración del entorno de ejecución

rm(list=ls()) # Eliminación de variables de entorno en memoria.
setwd(dirname(rstudioapi::getSourceEditorContext()$path)) # Configuración del directorio de trabajo.
path <- getwd() # Almacenamiento del path.
graphics.off() # Borra todos los graficos almacenados en memoria.

font_import(prompt = FALSE, pattern = "segoe")
loadfonts(device = "win")

###############################################################################
# 1. Lectura del dataset

url <- "https://raw.githubusercontent.com/albertofalco/attrition_prediction/main/data/HRDataset_v14.csv"
df <- read.csv(url, header = TRUE, sep = ",")

###############################################################################
# 2. Preprocesamiento

# Estructura del dataframe.
cat(str(df))

# Se convierte campo DOB (Date of birth) y se obtiene la edad correspondiente.
base_date <- as.Date("10/01/23", format = "%m/%d/%y")
df$DOB <- as.Date(df$DOB, format = "%m/%d/%y")
convert_dates <- function(date){
  if (as.integer(strftime(date, "%Y")) >= as.integer(strftime(base_date, "%Y"))) {
    dp <- as.POSIXlt(date)
    dp$year <- dp$year - 100
    date <- as.Date(dp)
  }
  return(as.Date(date))
}
df$Age <- as.numeric(base_date - as.Date(sapply(X = df$DOB, FUN = convert_dates))) / 365

# Se compara los campos DateofHire y DateofTermination y se obtiene la antiguedad correspondiente o la permanencia laboral, según correpsponda.
seniority_calc <- function(hire_date, termination_date){
  d1 <- as.Date(hire_date, format = "%m/%d/%Y")
  d2 <- as.Date(termination_date, format = "%m/%d/%Y")
  if(is.na(d2)){
    s <- base_date - d1
  }
  else{
    s <- d2 - d1
  }
  return(s)
}
df$Seniority <- as.numeric(mapply(FUN = seniority_calc, df$DateofHire, df$DateofTermination)) / 365

# Transformación de la variable EmploymentStatus.
df$EmploymentStatus <- as.integer(sapply(X = df$EmploymentStatus, FUN = function(x){if(x=="Voluntarily Terminated")return(1)else{return(0)}}))

# Estructura del dataframe.
cat(str(df))

###############################################################################
# 3. Calidad de datos.

# Recuento de valores vacios por atributo.
df %>% summarise_all(list(~sum(is.na(.))))

# La variable ManagerID posee valores NA.
df$ManagerName[is.na(df$ManagerID)]
df$ManagerID[df$ManagerName == "Webster Butler"]

# La variable posee ID = NA cuando el manager coincide con "Webster Butler". Pero su ID es 39. Corresponde corregir las desviaciones.
df$ManagerID[df$ManagerName == "Webster Butler"] <- 39
sum(is.na(df$ManagerID))
df %>% summarise_all(list(~sum(is.na(.))))

# Verificación de campos vacíos ("").
df %>% summarise_all(list(~sum(.=="")))
df$DOB == ""
df$DateofTermination == ""

# Sin perjuicio de DOB y DateofTermination que ya fueron analizadas previamente, no se detectan valores "".

###############################################################################
# 4. Conversión de variables a formato "factor".

# Columnas a transformar.
columns <- c("MarriedID", "MaritalStatusID", "GenderID", "EmpStatusID", "DeptID", "PerfScoreID", 
             "FromDiversityJobFairID", "Termd", "PositionID", "State", "Zip", "Sex", "MaritalDesc", 
             "CitizenDesc", "HispanicLatino", "RaceDesc", "ManagerID", "RecruitmentSource", "PerformanceScore"
             )
for(column in columns){
  df[[column]] <- as.factor(df[[column]])
}

# Estructura del dataframe.
cat(str(df))

###############################################################################
# 5. Analisis exploratorio inicial

# 5.1. Distribución de atributos

# Graficos de barras.
bar_plotter <- function(df, var){
  df_count <- df %>% group_by(EmploymentStatus, !!sym(var)) %>% summarise(n = n()) %>% mutate(freq = n / sum(n))
  p <- ggplot(df_count, aes(x = EmploymentStatus, y = freq, fill = factor(!!sym(var)))) +
    geom_col(position = "dodge") +
    labs(x = "EmploymentStatus", y = "Recuento", fill = var) +
    theme_light(base_family = "serif") +
    scale_fill_manual(values = c("#329932", "#6666ff", "#ff4c4c", "#ffc04c"))
  return(p)
}

# Obtención de gráficos.
p1 <- bar_plotter(df, "MarriedID")
p2 <- bar_plotter(df, "GenderID")
p3 <- bar_plotter(df, "PerfScoreID")
p4 <- bar_plotter(df, "CitizenDesc")
grid.arrange(p1, p2, p3, p4, nrow = 2)

# Histogramas.
hist_plotter <- function(df, var){
  p <- ggplot(data = df, mapping = aes(x = !!sym(var), fill = factor(EmploymentStatus))) +
          geom_histogram(bins = 20, alpha = 0.85) +
          theme_light(base_family = "serif") +
          scale_fill_manual(values = c("#329932", "#6666ff", "#ff4c4c", "#ffc04c"))
  return(p)
}

# Obtención de gráficos.
p1 <- hist_plotter(df, "Salary")
p2 <- hist_plotter(df, "Age")
p3 <- hist_plotter(df, "Absences")
p4 <- hist_plotter(df, "Seniority")
grid.arrange(p1, p2, p3, p4, nrow = 2)

# 5.2. Correlaciones - Test de correlaciones

# Variables explicativas categoricas.
corr_test <- function(df, x, y){
  options(warn = -1)
  # cat("Contingency table for ", x, " & ", y, ":\n\n")
  # tbl <- table(df[[x]], df[[y]], dnn = c(x, y))
  test <- chisq.test(df[[x]], df[[y]])
  # print(tbl)
  # print(test)
  s <- "X-squared = {round(test$statistic, 4)}, df = {test$parameter}, p-value = {round(test$p.value, 4)}\n\n"
  print(glue("Pearson's Chi-squared test with Yates' continuity correction"))
  print(glue("Variables: {x} & {y}\n"))
  print(glue(s))
  options(warn = TRUE)
  return(test)
}

# Columnas a evaluar.
columns_to_test <- c("MarriedID", "MaritalStatusID", "GenderID", "DeptID", "PerfScoreID")

# Variables explicativas categóricas.
for (column in columns_to_test){
  corr_test(df, column, "EmploymentStatus")
}

# Variables explicativas numéricas.
corr_test_num <- function(df, x, y){
  options(warn = -1)
  test <- cor.test(df[[x]], df[[y]])
  s <- "{test$method}
         Variables: {x} & {y}
         t = {round(test$statistic, 4)}, df = {test$parameter}, p-value = {round(test$p.value, 4)} \n\n"
  print(glue(s))
  options(warn = TRUE)
  return(test)
}

# Columnas a evaluar.
columns_to_test <- c("Salary", "Absences", "Age", "Seniority")

# Variables explicativas categóricas.
for (column in columns_to_test){
  corr_test_num(df, column, "EmploymentStatus")
}

###############################################################################
# 6. Modelado

# Función para la obtención de matrices de confusión.
conf_matrix <- function(true_values, prob_values){
  # Obtencion de la matriz de confusion.
  cm <- confusionMatrix(data = prob_values, reference = true_values, positive = "1")
  # Output de la matriz de confusion.
  return(cm)
}

# Función para la obtencion de curva ROC.
roc_curve <- function(true_values, prob_values){
  prediobj <- prediction(predictions = prob_values, labels = true_values)
  perf <-  performance(prediobj, "tpr","fpr") # Sensitividad vs Especificidad
  auc <- as.numeric(performance(prediobj,"auc")@y.values)
  par(family="Segoe UI Light")  
  plot(perf,
       main = "Curva ROC",
       xlab="1 - Especificidad (Tasa de falsos positivos)", 
       ylab="Sensibilidad (Tasa de verdaderos positivos)",
       col="#6666ff",
       lwd=2)
  grid()
  abline(a=0,b=1,col="#329932",lty=2)
  legend("bottomright",legend=paste(" AUC =",round(auc,4)))
  p <- recordPlot()
  graphics.off()
  return(list("p" = p, "auc" = auc))
}

# Función para el diseño del modelo y obtención de métricas.
model_pipeline <- function(data, y, f){
  # Funciones link.
  links <- list(gaussian(link = "identity"), binomial(link = "logit"), binomial(link = "probit"))
  models <- list()
  
  for (link in links){
    link_name <- link$link
    # Definición del modelo sin variables explicativas.
    mod <- glm(formula = f, data = data, family = link)
    models[[link_name]][["model"]] <- mod
    # Resumen del modelo.
    models[[link_name]][["summary"]] <- summary(mod)
    # Obtención de los coeficientes de determinación.
    models[[link_name]][["R2"]] <- round(DescTools::PseudoR2(mod, which = "all"), 4)
    # Conversion de valores reales y predicciones a factor.
    true_labels <- as.factor(y)
    predicted_labels <- factor(as.vector(ifelse(mod$fitted.values > 0.5, 1, 0)), levels = levels(true_labels))
    probabilities <- mod$fitted.values
    models[[link_name]][["true_labels"]] <- true_labels
    models[[link_name]][["predicted_labels"]] <- predicted_labels
    models[[link_name]][["probabilities"]] <- probabilities
    # Obtencion de la matriz de confusión. 
    cm <- conf_matrix(true_values = true_labels, prob_values = predicted_labels)
    models[[link_name]][["confusion_matrix"]] <- cm
    # Obtención de la curva ROC y valor AUC.
    roc <- roc_curve(true_values = true_labels, prob_values = probabilities)
    models[[link_name]][["roc_auc"]] <- roc
  }
  return(models)
}

# Función para la iteración por combinaciones de atributos.
formula_iterator <- function(df, y, attributes, return_pvalues = TRUE){
  models_list <- list()
  iterator <- 0
  for (att in attributes){
    iterator <- iterator + 1
    f_index <- paste("f_", iterator, sep = "")
    f <- paste("EmploymentStatus ~", att)
    mod <- model_pipeline(data = df, y = y, f = as.formula(f))
    mod_base <- model_pipeline(data = df, y = y, f = EmploymentStatus ~ 1)
    models_list[[f_index]] <- list()
    models_list[[f_index]][["f"]] <- f
    models_list[[f_index]][["mod"]] <- mod
    cat("Formula:", f, "\n\n")
    if(return_pvalues == TRUE){
      summary_df <- data.frame(
        "MLP" = c("Deviance" = mod$identity$summary$deviance,
                  "AIC" = mod$identity$summary$aic,
                  "AUC" = mod$identity$roc_auc$auc,
                  mod$identity$summary$coefficients[,4],
                  "Global" = lmtest::lrtest(mod$identity$model, mod_base$identity$model)[["Pr(>Chisq)"]][2]
        ),
        "Logit" = c("Deviance" = mod$logit$summary$deviance,
                    "AIC" = mod$logit$summary$aic,
                    "AUC" = mod$logit$roc_auc$auc,
                    mod$logit$summary$coefficients[,4],
                    "Global" = lmtest::lrtest(mod$logit$model, mod_base$logit$model)[["Pr(>Chisq)"]][2]
        ),
        "Probit" = c("Deviance" = mod$probit$summary$deviance,
                     "AIC" = mod$probit$summary$aic,
                     "AUC" = mod$probit$roc_auc$auc,
                     mod$probit$summary$coefficients[,4],
                     "Global" = lmtest::lrtest(mod$probit$model, mod_base$probit$model)[["Pr(>Chisq)"]][2]
        )
      )     
    }
    else{
      summary_df <- data.frame(
        "MLP" = c("Deviance" = mod$identity$summary$deviance,
                  "AIC" = mod$identity$summary$aic,
                  "AUC" = mod$identity$roc_auc$auc,
                  "Global" = lmtest::lrtest(mod$identity$model, mod_base$identity$model)[["Pr(>Chisq)"]][2]
        ),
        "Logit" = c("Deviance" = mod$logit$summary$deviance,
                    "AIC" = mod$logit$summary$aic,
                    "AUC" = mod$logit$roc_auc$auc,
                    "Global" = lmtest::lrtest(mod$logit$model, mod_base$logit$model)[["Pr(>Chisq)"]][2]
        ),
        "Probit" = c("Deviance" = mod$probit$summary$deviance,
                     "AIC" = mod$probit$summary$aic,
                     "AUC" = mod$probit$roc_auc$auc,
                     "Global" = lmtest::lrtest(mod$probit$model, mod_base$probit$model)[["Pr(>Chisq)"]][2]
        )
      )     
    }
    models_list[[f_index]][["summary_df"]] <- round(summary_df, 4)
    print(round(summary_df, 4))
    cat("\n")
    # cat("Devianzas (MLP, Logit, Probit):", mod$identity$summary$deviance, mod$logit$summary$deviance, mod$probit$summary$deviance, "\n")
    # cat("AICs (MLP, Logit, Probit):", mod$identity$summary$aic, mod$logit$summary$aic, mod$probit$summary$aic, "\n")
    # cat("AUCs (MLP, Logit, Probit):", mod$identity$roc_auc$auc, mod$logit$roc_auc$auc, mod$probit$roc_auc$auc, "\n\n")
  }
  return(models_list)
}

# Función para consolidación de metricas.
obtain_metrics <- function(models, metric, lower_margin, reference, padj){
  df <- data.frame(matrix(ncol = 4, nrow = 0))
  model_types <- c("MLP", "Logit", "Probit")
  formulas <- c()
  for(model in models){
    new_row <- c(model$summary_df[metric,model_types[1]],
                 model$summary_df[metric,model_types[2]],
                 model$summary_df[metric,model_types[3]]
    )
    formula_name <- sub(".*~\\s*", "", model$f)
    formulas <- c(formulas, formula_name)
    df <- rbind(df, new_row)
  }
  colnames(df) <- model_types
  m <- t(as.matrix(df))
  colnames(m) <- formulas
  f_labels <- vector(mode = "character", length = 54)
  f_labels[c(seq(2, 54, 3))] <- formulas
  par(mar = c(lower_margin,5,5,2), family = "Segoe UI Light", cex = 0.8)
  p <- barplot(m,
               horiz = FALSE,
               col = c("#ffc04c", "#329932", "#6666ff"),
               main = metric,
               xlab = NULL,
               ylab = paste(metric, reference),
               beside = TRUE,
               border = NA,
               las = 2,
               xaxt="n",
               legend = T)
  text(p, par("usr")[3], cex = 0.9, adj = 1, labels = f_labels, xpd=TRUE, srt=45)
  mtext("EmploymentStatus ~ ...", side = 1, line = 2, padj = padj)
  return(p)
}

###############################################################################
# 7. Iteraciones

# Primera iteracion para obtener el modelo base.
attributes <- c(1)
models_base <- formula_iterator(df = df, y = df$EmploymentStatus, attributes = attributes)

# Segunda iteración con una variable regresora, a efectos de evaluar la explicabilidad de los modelos.
attributes <- c("MarriedID", "MaritalStatusID", "GenderID", "DeptID", "PerfScoreID", "Salary",
                "PositionID", "HispanicLatino", "RaceDesc", "ManagerID", "RecruitmentSource",
                "EngagementSurvey", "EmpSatisfaction", "SpecialProjectsCount", "DaysLateLast30",
                "Absences", "Age", "Seniority")
models_1 <- formula_iterator(df = df, y = df$EmploymentStatus, attributes = attributes)

# Obtención de graficas.
obtain_metrics(models_1, "Deviance", 12, "(Lower is better)", 5)
obtain_metrics(models_1, "AIC", 12, "(Lower is better)", 5)
obtain_metrics(models_1, "AUC", 12, "(Greater is better)", 5)
obtain_metrics(models_1, "Global", 12, "(Lower is better)", 5)

# Tercera iteración sobre las mejores 5 variables explicativas obtenidas y la combinatoria de ellas de a 2.
attributes_elem <- c("Seniority", "ManagerID", "RecruitmentSource", "PositionID", "MaritalStatusID", "Age", "Salary")
attributes <- apply(combn(attributes_elem, m = 2), 2, paste, collapse = " + ")
models_2 <- formula_iterator(df = df, y = df$EmploymentStatus, attributes = attributes)

# Obtención de graficas.
obtain_metrics(models_2, "Deviance", 12, "(Lower is better)", 10)
obtain_metrics(models_2, "AIC", 12, "(Lower is better)", 10)
obtain_metrics(models_2, "AUC", 12, "(Greater is better)", 10)
obtain_metrics(models_2, "Global", 12, "(Lower is better)", 10)

# Cuarta iteración sobre las mejores 5 variables explicativas obtenidas y la combinatoria de ellas de a 3.
attributes_elem <- c("Seniority", "ManagerID", "RecruitmentSource", "PositionID", "MaritalStatusID", "Age", "Salary")
attributes <- apply(combn(attributes_elem, m = 3), 2, paste, collapse = " + ")
models_3 <- formula_iterator(df = df, y = df$EmploymentStatus, attributes = attributes)

# Quinta iteración con variables Seniority y RecruitmentSource permanentes.
attributes <- c("Seniority + RecruitmentSource + ManagerID", 
                "Seniority + RecruitmentSource + PositionID", 
                "Seniority + RecruitmentSource + MaritalStatusID", 
                "Seniority + RecruitmentSource + Age", 
                "Seniority + RecruitmentSource + Salary")
models_4 <- formula_iterator(df = df, y = df$EmploymentStatus, attributes = attributes)

# Sexta iteración con variables Seniority y RecruitmentSource permanentes.
attributes <- c("Seniority + RecruitmentSource + ManagerID + PositionID", 
                "Seniority + RecruitmentSource + ManagerID + PositionID + MaritalStatusID",
                "Seniority + RecruitmentSource + ManagerID + PositionID + Age",
                "Seniority + RecruitmentSource + ManagerID + PositionID + Salary")
models_5 <- formula_iterator(df = df, y = df$EmploymentStatus, attributes = attributes, return_pvalues = FALSE)

# Última iteración con todos los atributos.
attributes <- c("Seniority + RecruitmentSource + ManagerID + PositionID + MaritalStatusID + Age + Salary")
models_6 <- formula_iterator(df = df, y = df$EmploymentStatus, attributes = attributes, return_pvalues = TRUE)

###############################################################################
# 7. Validación de Hipotesis 1: Significación de Age.

# Aplicación de modelos para evaluar la significación de Age.
attributes <- c("Seniority + RecruitmentSource + ManagerID + PositionID + MaritalStatusID + Salary",
                "Seniority + RecruitmentSource + ManagerID + PositionID + MaritalStatusID + Age + Salary")
models_age <- formula_iterator(df = df, y = df$EmploymentStatus, attributes = attributes, return_pvalues = TRUE)

# Impresión de curvas ROC.
models_age$f_1$mod$logit$roc_auc
models_age$f_2$mod$probit$roc_auc

# Prueba de hipotesis de la razon de verosimilitud entre modelos.
for (i in 1:length(models_age$f_1$mod)){
  m1 <- models_age[["f_1"]][["mod"]][[i]][["model"]]
  m2 <- models_age[["f_2"]][["mod"]][[i]][["model"]]
  lr_test <- lmtest::lrtest(m2, m1)
  cat(paste("Funcion de link:", names(models_age[["f_1"]][["mod"]])[i], "\n"))
  cat(paste("Chi Squared:", lr_test$Chisq[2], "\n"))
  cat(paste("P-Value:", lr_test$`Pr(>Chisq)`[2], "\n\n"))
}

###############################################################################
# 8. Validación de Hipotesis 2: Significación de Seniority.

# Aplicación de modelos para evaluar la significación de Seniority
attributes <- c("RecruitmentSource + ManagerID + PositionID + MaritalStatusID + Age + Salary",
                "Seniority + RecruitmentSource + ManagerID + PositionID + MaritalStatusID + Age + Salary")
models_seniority <- formula_iterator(df = df, y = df$EmploymentStatus, attributes = attributes, return_pvalues = TRUE)

# Impresión de curvas ROC.
models_seniority$f_1$mod$logit$roc_auc
models_seniority$f_2$mod$probit$roc_auc

# Prueba de hipotesis de la razon de verosimilitud entre modelos.
for (i in 1:length(models_seniority$f_1$mod)){
  m1 <- models_seniority[["f_1"]][["mod"]][[i]][["model"]]
  m2 <- models_seniority[["f_2"]][["mod"]][[i]][["model"]]
  lr_test <- lmtest::lrtest(m2, m1)
  cat(paste("Funcion de link:", names(models_seniority[["f_1"]][["mod"]])[i], "\n"))
  cat(paste("Chi Squared:", lr_test$Chisq[2], "\n"))
  cat(paste("P-Value:", lr_test$`Pr(>Chisq)`[2], "\n\n"))
}

###############################################################################
# 9. Validación de Hipotesis 3: Perfomance del modelo logit.

# Definicion de atributos del modelo.
attributes <- c("Seniority + RecruitmentSource",
                "Seniority + RecruitmentSource + ManagerID + PositionID",
                "Seniority + RecruitmentSource + ManagerID + PositionID + MaritalStatusID + Age + Salary")
models_h3 <- formula_iterator(df = df, y = df$EmploymentStatus, attributes = attributes, return_pvalues = FALSE)

# Prueba de hipotesis de la razon de verosimilitud entre modelos.
for (model in models_h3){
  lr_test_1 <- lmtest::lrtest(model$mod$logit$model, model$mod$identity$model)
  lr_test_2 <- lmtest::lrtest(model$mod$logit$model, model$mod$probit$model)
  cat(paste("Fórmula:", model$f))
  print(model$summary_df)
  cat(paste("Logit vs. identity - Chi Squared:", lr_test_1$Chisq[2], "\n"))
  cat(paste("Logit vs. identity - P-Value:", lr_test_1$`Pr(>Chisq)`[2], "\n"))
  cat(paste("Logit vs. probit - Chi Squared:", lr_test_2$Chisq[2], "\n"))
  cat(paste("Logit vs. probit - P-Value:", lr_test_2$`Pr(>Chisq)`[2], "\n\n"))
}

###############################################################################
# 10. Interpretación de coeficientes.

# Definicion de atributos del modelo.
attributes <- c("Seniority + RecruitmentSource + ManagerID",
                "Seniority + RecruitmentSource + PositionID")
models_int <- formula_iterator(df = df, y = df$EmploymentStatus, attributes = attributes, return_pvalues = FALSE)

# Impresión de formulas y coeficientes.
print(models_int$f_1$f)
print(data.frame("Identity" = models_int$f_1$mod$identity$summary$coefficients[,1],
                 "Logit" = models_int$f_1$mod$logit$summary$coefficients[,1],
                 "Probit" = models_int$f_1$mod$probit$summary$coefficients[,1]
                 )
      )

print(models_int$f_2$f)
print(data.frame("Identity" = models_int$f_2$mod$identity$summary$coefficients[,1],
                 "Logit" = models_int$f_2$mod$logit$summary$coefficients[,1],
                 "Probit" = models_int$f_2$mod$probit$summary$coefficients[,1]
)
)
