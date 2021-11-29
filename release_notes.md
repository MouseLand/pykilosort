# 1.2
## 1.2.1
-   IBL pre-proc: add the channel removal code to the computing of the whitening matrix 

## 1.2.0 alpha02
-   fix bug for the last template (Kush)

## 1.2.0 alpha01 
-   ibllib > 2.5.0 pre-processing (destriping)
    -   channel rejection and interpolation before destriping
    -   uses the pykilosort parameters for the high-pass filter
    -   multi-processing version of the destriping
-   QC:
    -   destriping outputs the RMS of each batch after pre-processing
    -   outputs

# 1.1
-   add pre-processing within pykilosort
-   whitening is optional and set as a parameter

# 1.0
## 1.0.1 2021-08-08
-   output the drift matrix in Alf format
## 1.0.2 2021-08-31
-   attempt to fix bugs introduced by chronic recordings that reduce amount of detected spikes
