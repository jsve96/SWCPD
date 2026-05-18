library(ocp)
library(jsonlite)

zip_file <- paste(dirname(getwd()),"/datasets/MNISTSeq.zip",sep="")

json_file <- unzip(zip_file, exdir = tempdir())

data <- fromJSON(json_file)



lol <- list()
id <- 0
for (subdata in data){
  results <- list()
  datapts<-subdata$data
  print("CPS")
  start_time <- Sys.time()
  fit <- onlineCPD(
    datapts = datapts,                     # Your MNIST-like data (matrix with 784 columns)
    oCPD = NULL,                           # No previous model
    missPts = "none",                      # No missing points handling
    hazard_func = function(x, lambda) { 
      const_hazard(x, lambda = 100)        # Constant hazard function with lambda = 200
    },
    probModel = list("g"),                 # Gaussian observation model
    init_params = list(list(
      m = rep(0.3, 784),                   # Mean vector of length 784
      k = 0.01,                            # Scalar confidence in the mean
      a = 0.01,                               # Scalar shape parameter for variance
      b = 1e-4)),                           # Scalar scale parameter for variance
    
    multivariate = TRUE,                   # Enable multivariate mode
    cpthreshold = 0.5,                     # Threshold for detecting changepoints
    truncRlim = 10^(-4),                   # Limit for truncating probabilities
    minRlength = 1,                        # Minimum run length
    maxRlength = 10^4,                     # Maximum run length
    minsep = 1,                            # Minimum separation for changepoints
    maxsep = 10^4,                         # Maximum separation for changepoints
    timing = FALSE,                        # Disable timing output
    getR = FALSE,                          # Don't return posterior probabilities
    optionalOutputs = FALSE,               # Disable optional outputs
    printupdates = FALSE                   # Don't print updates
  )
  end_time <- Sys.time()
  print(end_time - start_time)
  print(fit$changepoint_lists$maxCPs)
  id_c <- as.character(id)
  print(id_c)
  results[['CPs']] <- fit$changepoint_lists$maxCPs
  results[['RunTime']] <- as.numeric(end_time-start_time)
  lol[[id_c]] <- results#fit$changepoint_lists$maxCPs
  id <- id+1
}

lol

write(toJSON(lol, pretty = TRUE), file = "MNIST_BOCPD_new.json")
