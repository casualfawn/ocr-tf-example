library(dplyr)
library(mongolite)
library(reshape2)
library(tidyverse)
#---mongo
cluster = 'mongodb_cluster_example'
url_path = 'example_url_path'
exampledb <- mongo(collection = "example", # Data Table
                db = "Digit_Data", # DataBase
                url = url_path, 
                verbose = TRUE)

#--- load python output and reorder values -----
digit_value1df <- read.csv('/home/username/Project Folder/digit_value1/digit_value1.csv')
digit_value1df <- digit_value1df[-1]
names(digit_value1df)[1] <- 'digit_value1'
digit_value1df$digit_value1 <- rev(digit_value1df$digit_value1)
digit_value1df

digit_value1df <- digit_value1df %>%
  mutate(digit_value1df,
         rn = ceiling(row_number() / 4)) %>% 
  group_by(rn) %>% 
  summarize(vlist = paste0(digit_value1, collapse = ''))
digit_value1df <- select(digit_value1df, vlist)
names(digit_value1df)[1] = 'digit_value1'
digit_value1df$digit_value1 <- sub("(.{1})(.*)", "\\1.\\2", digit_value1df$digit_value1 )
digit_value1df$digit_value1 <- as.numeric(digit_value1df$digit_value1)
digit_value1df

digit_value2df <- read.csv('/home/username/Project Folder/digit_value2/digit_value2.csv')
digit_value2df <- digit_value2df[-1]
names(digit_value2df)[1] = 'digit_value2'
digit_value2df
digit_value2df <- digit_value2df %>%
  mutate(digit_value2df,
         rn = ceiling(row_number() / 3)) %>% 
  group_by(rn) %>% 
  summarize(vlist = paste0(digit_value2, collapse = ''))
digit_value2df <- select(digit_value2df, vlist)
names(digit_value2df)[1] = 'digit_value2'

digit_value2df$digit_value2 = as.character(digit_value2df$digit_value2)
digit_value2df$digit_value2 <- sub("(.{2})(.*)", "\\1.\\2", digit_value2df$digit_value2)
digit_value2df$digit_value2 <- as.numeric(digit_value2df$digit_value2)

digit_value3df <- read.csv('/home/username/Project Folder/digit_value3/digit_value3.csv')

digit_value3df <- digit_value3df[-1]
names(digit_value3df)[1] = 'digit_value3'
digit_value3df <- digit_value3df %>%
  mutate(digit_value3df,
         rn = ceiling(row_number() / 3)) %>% 
  group_by(rn) %>% 
  summarize(vlist = paste0(digit_value3, collapse = ''))
digit_value3df <- select(digit_value3df, vlist)
names(digit_value3df)[1] = 'digit_value3'
digit_value3df$digit_value3 = as.character(digit_value3df$digit_value3)
digit_value3df$digit_value3 <- sub("(.{1})(.*)", "\\1.\\2", digit_value3df$digit_value3)
digit_value3df$digit_value3 <- as.numeric(digit_value3df$digit_value3)



digit_value4df <- read.csv('/home/username/Project Folder/digit_value4/digit_value4.csv')
digit_value4df <- digit_value4df[-1]
names(digit_value4df)[1] = 'digit_value4'

digit_value4df <- digit_value4df %>%
  mutate(digit_value4df,
         rn = ceiling(row_number() / 3)) %>% 
  group_by(rn) %>% 
  summarize(vlist = paste0(digit_value4ature, collapse = ''))

digit_value4df <- select(digit_value4df, vlist)
digit_value4df$vlist <- sub("(.{2})(.*)", "\\1.\\2", digit_value4df$vlist)
digit_value4df$vlist <- as.numeric(digit_value4df$vlist)
names(digit_value4df)[1] = 'digit_value4'

digit_value5df <- read.csv('/home/username/Project Folder/digit_value5/digit_value5.csv')
digit_value5df <- digit_value5df[-1]
names(digit_value5df)[1] = 'digit_value5'

digit_value5df <- digit_value5df %>%
  mutate(digit_value5df,
         rn = ceiling(row_number() / 3)) %>% 
  group_by(rn) %>% 
  summarize(vlist = paste0(digit_value5, collapse = ''))
digit_value5df <- select(digit_value5df, vlist)
names(digit_value5df)[1] = 'digit_value5'
digit_value5df$digit_value5 <- sub("(.{2})(.*)", "\\1.\\2", digit_value5df$digit_value5)
digit_value5df$digit_value5 <- as.numeric(digit_value5df$digit_value5)

all_digit_values_df = cbind(digit_value1df, digit_value2df, digit_value3df, digit_value4df,digit_value5df)
all_digit_values_df$timestamp = Sys.time()
all_digit_values_df$digit_value5 = 0
all_digit_values_df
write.csv(all_digit_values_df, '/home/username/Project Folder/all_digit_values_df.csv')


#-- mongolite upload Project Folder data -----
exampledb$insert(all_digit_values_df)
exampledb$disconnect()


rm(list=ls())
gc()
