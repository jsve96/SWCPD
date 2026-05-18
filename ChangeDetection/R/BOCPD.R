library(ocp)



IDs = list(10,14,7,182,225,19,185,33,36,87,88,210,11,20,23,243,247,91,95,96,100,141,91,95,245)
lol <- list()

current_dir <- getwd()

datasetpath <- paste(dirname(getwd()),"/datasets/has2023DataR/HASC_TS_",sep="")

for (id in IDs){
  print(id)
  
  datasetpath <-  paste(paste(datasetpath,id,sep = ""),'.csv',sep="")
  X<-  read.csv(datasetpath)
  datapts <- X[,-1]
  print('Start')
  start_time <- Sys.time()
  fit<- onlineCPD(datapts, oCPD = NULL, missPts = "none",
                  hazard_func = function(x, lambda) { const_hazard(x, lambda = 100)
                  }, probModel = list("g"), init_params = list(list(m = 0, k = 10, a
                                                                    = 0.1, b = 10)), multivariate = TRUE, cpthreshold = 0.5,
                  truncRlim = 10^(-4), minRlength = 1,
                  maxRlength = 10^4, minsep = 1, maxsep = 10^4, timing = FALSE,
                  getR = FALSE, optionalOutputs = FALSE, printupdates = FALSE)
  end_time <- Sys.time()
  print(end_time - start_time)
  print(fit$changepoint_lists$maxCPs)
  id_c <- as.character(id)
  print(id_c)
  lol[[id_c]] <- fit$changepoint_lists$maxCPs
  print(lol)
  #hashmap$id_c <- res$estimates
}

library(jsonlite)

# Save the list of lists as JSON
write(toJSON(lol, pretty = TRUE), file = "HASC_BOCPDL100m0k10a01b10.json")


### Occupancy with normalization
datapath <-  paste(dirname(getwd()),"/datasets/Occupancy/Occupancy.csv",sep="")


X <- read.csv(datasetpath)
datapts <- X[,-1]

z_score_normalized <- t(apply(datapts, 1, function(row) (row - mean(row)) / sd(row)))

# Convert back to a data frame
z_score_normalized <- as.data.frame(z_score_normalized)
colnames(z_score_normalized) <- colnames(datapts)
rownames(z_score_normalized) <- rownames(datapts)

print('Start')
start_time <- Sys.time()
fit <- onlineCPD(
  datapts = z_score_normalized,                     # Your MNIST-like data (matrix with 784 columns)
  oCPD = NULL,                           # No previous model
  missPts = "none",                      # No missing points handling
  hazard_func = function(x, lambda) { 
    const_hazard(x, lambda = 100)        # Constant hazard function with lambda = 200
  },
  probModel = list("g"),                 # Gaussian observation model
  init_params = list(list(
    m = rep(0.0, 4),                   # Mean vector of length 784
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

write(toJSON(fit$changepoint_lists$maxCPs, pretty = TRUE), file = "OccupancyBOCPD.json")
