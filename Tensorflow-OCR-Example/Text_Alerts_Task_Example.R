library(twilio)
library(dplyr)
library(lubridate)
library(purrr)
library(data.table)
library(mongolite)

Sys.setenv(TWILIO_SID = "") #your twilio sid and token
Sys.setenv(TWILIO_TOKEN = "")
from1 = '+'#your twilio number
receiver1 = ''
receiver2 = '+' #numbers to get alerts 

#---

value_1_low_threshold = 12
value_1_high_threshold = 13

value1df <- read.csv('/home/user/Project Folder/value1.csv')
value1df <- value1df[-1]
valuedf$timestamp <- Sys.time()
names(value1df)[1] = 'value1'
value1df
value1df <- value1df %>%
  mutate(value1df,
         rn = ceiling(row_number() / 3)) %>% 
  group_by(rn) %>% 
  summarize(vlist = paste0(value1, collapse = ''))
value1df <- select(value1df, vlist)
names(value1df)[1] = 'value1'

value1df$value1 = as.character(value1df$value1)
value1df$value1 <- sub("(.{2})(.*)", "\\1.\\2", value1df$value1)
value1df$value1 <- as.numeric(value1df$value1)
value1 <- value1df


if (value1[1] > value_1_high_threshold){
  tw_send_message(from = from1, to = receiver1, body = paste0('the value1 value as of '  %>%
                                                                    paste0(value1df$timestamp)  %>% paste0(' is above the threshold, you might want to check it ') %>% paste0('                                 Current value1 value is: ') %>% paste0(value1[1])))
  print('value1 is above threshold message will be sent abovethreshold')
} else if(value1[1] < value_1_low_threshold){
  tw_send_message(from = from1, to = receiver1, body = paste0('the value1 value as of ' %>% 
                                                                    paste0(value1df$timestamp)   %>% paste0(' is below the threshold, you might want to check it ') %>% paste0('                                 Current value1 value is: ') %>% paste0(value1[1])))
  print('value1 is below threshold message sent belowthreshold')
} else {
  print('nothingtovalue1here')
}

