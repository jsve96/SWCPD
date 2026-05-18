library(ecp)
library(hash)
library(jsonlite)




zip_file <- paste(dirname(getwd()),"/datasets/MNISTSeq.zip",sep="")

json_file <- unzip(zip_file, exdir = tempdir())

data <- fromJSON(json_file)


lol <- list()
id <- 0
for (subdata in data){
  results <-list()
  datapts<-subdata$data
  print("CPS")
  #print(subdata$labels)
  start_time <- Sys.time()
  res <- e.divisive(datapts,R=199,sig.lvl = 0.05)
  end_time <- Sys.time()
  print(res$estimates)
  print(end_time - start_time)
  id_c <- as.character(id)
  print(id_c)
  results[['CPs']] <- res$estimates
  results[['RunTime']] <- as.numeric(end_time - start_time)
  lol[[id_c]] <- results #res$estimates
  id <- id+1
   
  
}

lol
  
write(toJSON(lol, pretty = TRUE), file = "MNIST_Edivise_new.json")


