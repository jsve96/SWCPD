library(rulsif.ts)
library(hash)




current_dir <- getwd()

datasetpath <- paste(dirname(getwd()),"/datasets/has2023DataR/HASC_TS_",sep="")
print(datasetpath)
IDs = list(10,14,7,182,225,19,185,33,36,87,88,210,11,20,23,243,247,91,95,96,100,141,91,95,245)
lol <- list()
for (id in IDs){
  print(id)
  
  string = paste(paste(datasetpath,id,sep = ""),'.csv',sep="")
  X<-  read.csv(string)
  datapts <- X[,-1]
  print('Start')
  start_time <- Sys.time()
  res <- e.divisive(datapts,R=199,min.size = 500)
  end_time <- Sys.time()
  print(end_time - start_time)
  print(res$estimates)
  id_c <- as.character(id)
  print(id_c)
  lol[[id_c]] <- res$estimates
  print(lol)
  #hashmap$id_c <- res$estimates
}



library(jsonlite)

# Save the list of lists as JSON
write(toJSON(lol, pretty = TRUE), file = "HASC_EdiviseR199MINSIZE500.json")


# Occupancy

datapath <-  paste(dirname(getwd()),"/datasets/Occupancy/Occupancy.csv",sep="")

dirname <- "//wsl.localhost/Ubuntu/home/sven/coding/python/18229_High_Dimensional_Online__Supplementary Material/CODE_SWCPD/ChangeDetection/datasets"

datapath <-  paste(dirname,"/Occupancy/Occupancy.csv",sep="")


X <- read.csv(datapath)
datapts <- X[,-1]

my_matrix <- t(matrix(unlist(datapts), ncol = 4, byrow = TRUE))

for (i in c(1,2,3,4,5)){
  print('Start')
  start_time <- Sys.time()
  res <- ts_detect(my_matrix,window_size = 30)#e.divisive(datapts,R=30,sig.lvl=0.05,min.size = 400)
  end_time <- Sys.time()
  print(res$estimates)
  print(end_time-start_time)}



s <- c(rnorm(150, mean = 0), rnorm(150, mean = 5), rnorm(150, mean = 1))
s <- matrix(s, nrow = 1)
ts_detect(s)


do.call()
