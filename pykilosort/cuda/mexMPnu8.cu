const int  Nthreads = 1024, maxFR = 100000, NrankMax = 3, nmaxiter = 500, NchanMax = 32;

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	spaceFilter(const double *Params, const float *data, const float *U,
        const int *iC, const int *iW, float *dprod){

// <<<Nfilt, Nthreads>>>
// blockIdx = current filter/template
// blockDim = 1024 (max number of threads)
// threadIdx = used both to index channel (in synchronized portion)
// and time (in non-synchronized portion).
  volatile __shared__ float  sU[32*NrankMax];
  volatile __shared__ int iU[32];
  float x;
  int tid, bid, i,k, Nrank, Nchan, NT, Nfilt, NchanU;

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  NT      	= (int) Params[0];
  Nfilt    	= (int) Params[1];
  Nrank     = (int) Params[6];
  NchanU    = (int) Params[10];     //NchanNear in learnTemplates = 32
  Nchan     = (int) Params[9];

  if (tid<NchanU)
      iU[tid] = iC[tid + NchanU * iW[bid]];     //channels for this filter (given by iW)
  __syncthreads();

  //U is array of Nchan x Nfilt x Nrank, spatial PCs for each channel for each template
  //Populate sU array with these pcs to calculate projection of data onto them
  if(tid<NchanU*Nrank)
      sU[tid]= U[iU[tid%NchanU] + Nchan * bid + Nchan * Nfilt * (tid/NchanU)];

  //sU[tid]= U[tid%NchanU + NchanU * bid + NchanU * Nfilt * (tid/NchanU)];

  __syncthreads();

//with arrays populated, loop over timepoints in blocks of 1024.
//__syncthreads ensures the iU and SU arrays are filled before starting this loop
//dprod = NT x NFilt*Nrank, projections of each time point on 3 pcs of each
//spatial filter

    while (tid<NT){
        for (k=0;k<Nrank;k++){
            volatile float *pSU = &sU[NchanU*k];
            x = 0.0f;
            for(i=0;i<NchanU;i++)
                x += *pSU++ * data[tid + NT * iU[i]];
            dprod[tid + NT*(bid + k*Nfilt)] = x;
        }

        tid += blockDim.x;
        __syncthreads();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	spaceFilterUpdate(const double *Params, const float *data, const float *U, const bool *UtU,
        const int *iC, const int *iW, float *dprod,  const int *st, const int *id, const int *counter){
    volatile __shared__ float  sU[32*NrankMax];
    volatile __shared__ int iU[32];
    float x;
    int tid, bid, ind, nt0, i, t, k, Nrank, NT, Nfilt, NchanU, Nchan;

    tid 	  = threadIdx.x;
    bid 	  = blockIdx.x;
    NT        = (int) Params[0];
    Nfilt     = (int) Params[1];
    Nrank     = (int) Params[6];
    NchanU    = (int) Params[10];
    nt0       = (int) Params[4];
    Nchan     = (int) Params[9];

    //<<<Nfilt, 2*nt0-1>>>
    // just need to do this for all filters that have overlap with id[bid] and st[id]
    // as in spaceFilter, tid = threadIdx.x is first used to index over channels and pcs
    // then used to loop over time, now just from -nt0 to nt0 about the input spike time
    // tidx represents time, from -nt0 to nt0
    // tidy loops through all filters that have overlap

    if (tid<NchanU)
        iU[tid] = iC[tid + NchanU * iW[bid]];
    __syncthreads();

    if (tid<NchanU) {
        for (k=0;k<Nrank;k++)
            sU[tid + k * NchanU] = U[iU[tid] + Nchan*(bid + Nfilt * k)];
    }
    __syncthreads();

    //each block corresponds to a filter
    //loop over all new spikes checking for matches to current filter (bid)
    //dprod = NT
    for(ind=counter[1];ind<counter[0];ind++){
        if (UtU[id[ind] + Nfilt * bid]){
            t = st[ind] + tid - nt0;
            // if this is a hit, threads compute all time offsets
            if (t>=0 & t<NT){
                for (k=0;k<Nrank;k++){
                    volatile float *pSU = &sU[NchanU*k];
                    x = 0.0f;
                    for(i=0;i<NchanU;i++)
                        x += *pSU++ * (float)(data[t + NT * iU[i]]);
                    dprod[t + NT*(bid + k*Nfilt)] = x;
                }
            }            
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	spaceFilterUpdate_v2(const double *Params, const double *data, const float *U, const bool *UtU,
        const int *iC, const int *iW, float *dprod,  const int *st, const int *id, const int *counter){
    volatile __shared__ float  sU[32*NrankMax];
    volatile __shared__ int iU[32];
    float x;
    int tid, bid, ind, nt0, i, t, k, Nrank, NT, Nfilt, NchanU, Nchan;
    
    tid 	  = threadIdx.x;
    bid 	  = blockIdx.x;
    NT        = (int) Params[0];
    Nfilt     = (int) Params[1];
    Nrank     = (int) Params[6];
    NchanU    = (int) Params[10];
    nt0       = (int) Params[4];
    Nchan     = (int) Params[9];
    
    //<<<Nfilt, 2*nt0-1>>>
    // just need to do this for all filters that have overlap with id[bid] and st[id]
    // as in spaceFilter, tid = threadIdx.x is first used to index over channels and pcs
    // then used to loop over time, now just from -nt0 to nt0 about the input spike time
    // tidx represents time, from -nt0 to nt0
    // tidy loops through all filters that have overlap
    
    if (tid<NchanU)
        iU[tid] = iC[tid + NchanU * iW[bid]];
    __syncthreads();
    
    if (tid<NchanU) {
        for (k=0;k<Nrank;k++)
            sU[tid + k * NchanU] = U[iU[tid] + Nchan*(bid + Nfilt * k)];
    }
    __syncthreads();
    
    //each block corresponds to a filter
    //loop over all new spikes checking for matches to current filter (bid)
    //dprod = NT x Nfilt x Nrank
    for(ind=counter[1];ind<counter[0];ind++){
        if (UtU[id[ind] + Nfilt * bid]){
            t = st[ind] + tid - nt0;
            // if this is a hit, threads compute all time offsets
            if (t>=0 & t<NT){
                for (k=0;k<Nrank;k++){
                    volatile float *pSU = &sU[NchanU*k];
                    x = 0.0f;
                    for(i=0;i<NchanU;i++)
                        x += *pSU++ * (float)(data[t + NT * iU[i]]);
                    dprod[t + NT*(bid + k*Nfilt)] = x;
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	timeFilter(const double *Params, const float *data, const float *W,float *conv_sig){
  volatile __shared__ float  sW2[81*NrankMax], sW[81*NrankMax], sdata[(Nthreads+81)*NrankMax];
  float x;
  int tid, tid0, bid, i, nid, Nrank, NT, Nfilt, nt0, irank;

    tid   = threadIdx.x;
    bid   = blockIdx.x;
    NT    = (int) Params[0];
    Nfilt = (int) Params[1];
    Nrank = (int) Params[6];
    nt0   = (int) Params[4];
    irank = tid/nt0;

// <<<Nfilt, Nthreads>>>
// threadIdx.x used as index over pcs in temporal templates
// (num PCs * number of timepoints = Nrank * nt0)
// Applied to data that's already been through filtering with
// the spatial templates, input data has dim Nrank x NT x Nfilt

    if(tid<nt0*Nrank)
        sW[tid] = W[tid%nt0 + (bid + Nfilt * irank) * nt0];
    __syncthreads();

// aftter sync threads, threadIdx.x = index over time points
// only goes up to NT - Nthreads-nt0+1 to avoid the template running
// past the end of the time series data.

    tid0 = 0;
    while (tid0<NT-Nthreads-nt0+1){
	    if (tid<nt0*NrankMax){
            sdata[tid%nt0 + irank*(Nthreads+nt0)] =
			    data[tid0 + tid%nt0 + NT*(bid + Nfilt*irank)];
        }

        #pragma unroll 3
        for(nid=0;nid<Nrank;nid++){
            sdata[tid + nt0 + nid*(Nthreads+nt0)] = data[nt0 + tid0 + tid + NT*(bid + nid*Nfilt)];
	    }
	    __syncthreads();

	    x = 0.0f;
        for(nid=0;nid<Nrank;nid++){
            volatile float *pSW = &sW[nid*nt0];
            volatile float *pSD = &sdata[tid + nid*(Nthreads+nt0)];
 		    #pragma unroll 4
            for(i=0;i<nt0;i++)
                x += *pSW++ * *pSD++;
	    }
        conv_sig[tid0  + tid + NT*bid] = x;

        tid0 += Nthreads;
        __syncthreads();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	timeFilterUpdate(const double *Params, const float *data, const float *W,
        const bool *UtU, float *conv_sig, const int *st, const int *id, const int *counter){

  volatile __shared__ float  sW[81*NrankMax], sW2[81*NrankMax];
  float x;
  int tid, tid0, bid, t, k,ind, Nrank, NT, Nfilt, nt0;

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  NT      	= (int) Params[0];
  Nfilt    	= (int) Params[1];
  Nrank     = (int) Params[6];
  nt0       = (int) Params[4];
// <<<Nfilt, Nthreads>>>
// Same as timeFilter, except timepoints now limited to +/- nt0 about
// spike times assiged to filters that may overlap the current filter
// specified by bid. The matrix of potentially overlapping filters
// is given in UtU.

    if (tid<nt0){
        for (k=0;k<Nrank;k++)
            sW[tid + k*nt0] = W[tid + nt0*(bid + Nfilt*k)];
    }
    __syncthreads();

    for(ind=counter[1];ind<counter[0];ind++) {
        if (UtU[id[ind] + Nfilt * bid]){
            tid0 = st[ind] - nt0 + tid;
            if (tid0>=0 && tid0<NT-nt0){
                x = 0.0f;
                for (k=0;k<Nrank;k++){
                    for (t=0;t<nt0;t++)
                        x += sW[t + k*nt0] * data[t + tid0 + NT*(bid + Nfilt*k)];
                }
                conv_sig[tid0 + NT*bid] = x;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void  bestFilter(const double *Params, const float *data,
	const float *mu, float *err, float *eloss, int *ftype){

    float  Cf, Cbest, lam, b, a, Cnextbest;
    int tid, tid0, i, bid, NT, Nfilt, ibest = 0, nt0;

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  NT 		= (int) Params[0];
  Nfilt 	= (int) Params[1];
  lam 	    = (float) Params[7];
  nt0       = (int) Params[4];

// <<<NT/Ntrheads, Nthreads>>>
// loop over timepoints

  tid0 = tid + bid * blockDim.x;
  while (tid0<NT-nt0){
      Cbest = 0.0f;
      Cnextbest = 0.0f;

      for (i=0; i<Nfilt;i++){
          a = 1+ lam;
          b = max(0.0f, data[tid0 + NT * i]) + lam * mu[i];
          Cf =  b*b/a - lam * mu[i]*mu[i];
          
          //a = lam * lam + mu[i] * mu[i];
          //b = max(0.0f, data[tid0 + NT * i]);
          //Cf = -mu[i]*mu[i] + 2 * mu[i] * b + mu[i]*mu[i] * (b - mu[i])*(b - mu[i]) / a;
          
          if (Cf > Cbest + 1e-6){
              Cnextbest = Cbest;
              Cbest 	= Cf;
              ibest 	= i;
          }
          else
              if  (Cf > Cnextbest + 1e-6)
                    Cnextbest = Cf;
      }
      err[tid0] 	= Cbest;
      eloss[tid0] 	= Cbest - Cnextbest;
      ftype[tid0] 	= ibest;

      tid0 += blockDim.x * gridDim.x;
  }
}

// THIS UPDATE DOES NOT UPDATE ELOSS?
//////////////////////////////////////////////////////////////////////////////////////////
__global__ void  bestFilterUpdate(const double *Params, const float *data,
	const float *mu, float *err, float *eloss, int *ftype, const int *st, const int *id, const int *counter){

  float  Cf, Cbest, lam, b, a, Cnextbest;
  int tid,  ind, i,t, NT, Nfilt, ibest = 0, nt0;

  tid 		= threadIdx.x;
  NT 		= (int) Params[0];
  Nfilt 	= (int) Params[1];
  lam 	    = (float) Params[7];
  nt0       = (int) Params[4];
  
  // we only need to compute this at updated locations
  ind = counter[1] + blockIdx.x;
  
  if (ind<counter[0]){
      t = st[ind]-nt0 + tid;
      if (t>=0 && t<NT){
          Cbest = 0.0f;
          for (i=0; i<Nfilt;i++){
              a = 1+ lam;
              b = max(0.0f, data[t + NT * i]) + lam * mu[i];
              Cf =  b*b/a - lam * mu[i]*mu[i];
              
              //a = lam * lam + mu[i] * mu[i];
              //b = max(0.0f, data[t + NT * i]);              
              //Cf = -mu[i]*mu[i] + 2 * mu[i] * b + mu[i]*mu[i] * (b - mu[i])*(b - mu[i]) / a;
              
              if (Cf > Cbest + 1e-6){
                  Cnextbest = Cbest;
                  Cbest 	= Cf;
                  ibest 	= i;
              }
              else
                  if  (Cf > Cnextbest + 1e-6)
                      Cnextbest = Cf;
          }
          err[t] 	= Cbest;
          ftype[t] 	= ibest;
      }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	cleanup_spikes(const double *Params, const float *data,
        const float *mu, const float *err, const float *eloss, const int *ftype, int *st,
        int *id, float *x, float *y,  float *z, int *counter){

  volatile __shared__ float sdata[Nthreads+2*81+1];
  float err0, Th;
  int lockout, indx, tid, bid, NT, tid0,  j, id0, t0;
  bool flag=0;

  // <<<NT/Nthreads,Nthreads>>>
  lockout   = (int) Params[4] - 1; // Parms[4] = nt0
  tid 		= threadIdx.x;
  bid 		= blockIdx.x;

  NT      	= (int) Params[0];
  tid0 		= bid * blockDim.x ;
  Th 		= (float) Params[2];
  //lam 	    = (float) Params[7];

  while(tid0<NT-Nthreads-lockout+1){
      if (tid<2*lockout)
          sdata[tid] = err[tid0 + tid];
      sdata[tid+2*lockout] = err[2*lockout + tid0 + tid];

      __syncthreads();

      err0 = sdata[tid+lockout];
      if(err0>Th*Th){
          flag = 0;
          for(j=-lockout;j<=lockout;j++)
              if(sdata[tid+lockout+j]>err0){
                  flag = 1;
                  break;
              }
          if(flag==0){
              indx = atomicAdd(&counter[0], 1);
              if (indx<maxFR){
                  t0        = tid+lockout+tid0;
                  id0       = ftype[t0];
                  st[indx] = t0;
                  id[indx] = id0;
                  y[indx]  = data[t0 + NT * id0];

                  //a = 1+ lam;
                  //b = max(0.0f, data[t0 + NT * id0]) + lam * mu[id0];

                  x[indx] = sqrt(err0);
                  //x[indx]  = b/a;    // do I really need this here?
                  //x[indx]  = y[indx];
                  z[indx]  = eloss[t0];
              }
          }
      }

      tid0 += blockDim.x * gridDim.x;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	extractFEAT(const double *Params, const int *st, const int *id,
        const int *counter, const float *dout, const int *iList,
        const float *mu, float *d_feat){

    float rMax, Ci, Cf, lam;
    int t, tidx, tidy, Nblocks, NthreadsX, idF, bid, NT, ind, tcurr, Nnearest;

    tidx 		= threadIdx.x;
    tidy 		= threadIdx.y;

    bid 		= blockIdx.x;
    NT 		    = (int) Params[0];
    Nnearest 	= (int) Params[5];
    NthreadsX 	= blockDim.x;
    Nblocks     = gridDim.x;
    lam 	    = (float) Params[7];

    // each thread x does a nearby filter
    // each thread x combines with blocks to go through all new spikes
    ind = counter[1]+tidx + NthreadsX * bid;

    while(ind<counter[0]){
        tcurr = st[ind];
        rMax = 0.0f;
        idF = iList[tidy + Nnearest * id[ind]];

        for (t=-3;t<3;t++){
            Ci = dout[tcurr +t+ idF * NT] + lam/mu[idF];
            Cf = Ci / sqrt(lam/(mu[idF] * mu[idF]) + 1.0f);
            rMax = max(rMax, Cf);
        }
        d_feat[tidy + ind * Nnearest] = rMax;
        ind += NthreadsX * Nblocks;
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// subtract_spikes version using single precision arithemtic and atomic operations to 
// avoid thread interference when threading over spikes. This calculation is not 
// deterministic, due to the order dependence of operations in single precision.
__global__ void	subtract_spikes(const double *Params,  const int *st, 
        const int *id, const float *x, const int *counter, float *dataraw, 
        const float *W, const float *U){
  int nt0, tidx, tidy, k, NT, ind, Nchan, Nfilt, Nrank;
  float X;

  NT        = (int) Params[0];
  nt0       = (int) Params[4];
  Nchan     = (int) Params[9];
  Nfilt    	=   (int) Params[1];
  Nrank     = (int) Params[6];

  tidx 		= threadIdx.x;
  ind       = counter[1]+blockIdx.x;

  while(ind<counter[0]){
      tidy = threadIdx.y;

      while (tidy<Nchan){
          X = 0.0f;
          for (k=0;k<Nrank;k++)
              X += W[tidx + id[ind]* nt0 + nt0*Nfilt*k] * 
                      U[tidy + id[ind] * Nchan + Nchan*Nfilt*k];                        
          
          X = -x[ind]*X;
          atomicAdd(&dataraw[tidx + st[ind] + NT * tidy], X);          
          tidy += blockDim.y;
      }
      ind += gridDim.x;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
//subtractions from array of doubles
__global__ void	subtract_spikes_v4(const double *Params,  const int *st, 
        const int *id, const float *x, const int *counter, double *dataraw, 
        const float *W, const float *U){
  
  double X;
  int nt0, tidx, tidy, k, NT, ind, Nchan, Nfilt, Nrank;

  unsigned long long int* address_as_ull;
  unsigned long long int old, assumed;                              

  NT        = (int) Params[0];
  nt0       = (int) Params[4];
  Nchan     = (int) Params[9];
  Nfilt    	= (int) Params[1];
  Nrank     = (int) Params[6];
  
  tidx 		= threadIdx.x;
  ind       = counter[1]+blockIdx.x;
  
  while(ind<counter[0]){
      tidy = threadIdx.y;
      
      while (tidy<Nchan){
          X = 0.0;     
          for (k=0;k<Nrank;k++)
              X += (double)((W[tidx + id[ind]* nt0 + nt0*Nfilt*k])) * 
                      (double)((U[tidy + id[ind] * Nchan + Nchan*Nfilt*k])); 
          X = -(double)(x[ind]) * X;
           
          address_as_ull = (unsigned long long int*)(&dataraw[tidx + st[ind] + NT * tidy]);
          old = *address_as_ull;
          do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(X + __longlong_as_double(assumed)));                              
               // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
          } while (assumed != old);

          tidy += blockDim.y;
      }
      ind += gridDim.x;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
//convert raw data to doubles for subtractions
//Tested with converting from 1 to 1000 values per thread. Found converting
//one value per thread is fastest.
//Tested running with 1 to 100 blocks; speed plateaus between 10 and 100 blocks
__global__ void	convToDouble(const double *Params, const float *singleData, 
        double *doubleData ) {
            
  int Nelem  = (int)Params[0] * (int)Params[9];  // NT * Nchan
  int start  = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  while (start < Nelem) {     
      doubleData[start] = singleData[start];
      start += stride;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
//convert data back to single
//see performance notes in comments for convToDouble
__global__ void	convToSingle(const double *Params, const double *doubleData, 
        float *singleData ) {
            
  int Nelem  = (int)Params[0] * (int)Params[9];  // NT * Nchan
  int start  = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  
  while (start < Nelem) {     
      singleData[start] = __double2float_rz(doubleData[start]);
      start += stride;
  }
}


//////////////////////////////////////////////////////////////////////////////////////////
/// JIC threaded only over channels to avoid collisions at specific time points
/// NChan/16 blocks, 16 threads. If the order of spikes is fixed (by sorting before 
/// subtraction) this routine is engineered for deterministic calculations.
/// However, it is almost 2X slower than the standard "usually deterministic"
/// calculation using unordered spikes but double precision arithemetic
/// (substract_spikes_v4).
/// If a guaranteed deterministic calculation is required, enable this routine
/// using the ENSURE_DETERM compile switch. This will also require the
/// ENABLE_STABLE_MODE to be set to one
__global__ void	subtract_spikes_v2(const double *Params,  const int *st, const unsigned int *idx,
        const int *id, const float *x, const int *counter, float *dataraw,
        const float *W, const float *U){

  int nt0, k, NT, ind, Nchan, Nfilt, Nrank, currChan, currInd;

  NT        = (int) Params[0];
  nt0       = (int) Params[4];
  Nchan     = (int) Params[9];
  Nfilt    	= (int) Params[1];
  Nrank     = (int) Params[6];

  //Note that the spike times st0 are the best position of the start of the
  //temporal temeplate; actual peak will be nt0min samples away nt0min = 20
  //therefore, st(ind) has a possible range of 0 to NT - nt0 - 1

  //spikes will be subtracted in the order given by idx. In the host code,
  //this array is filled either with counter[1] to counter[0] or the
  //time sorted indicies of the spikes.

  // W dims = (nt0 x Nfilt x Nrank)
  //Indexing into W, need to got to nt0*(index of this template) + nt0*Nfilt*index of pc

  // U dims = (NChan x Nfilt xNrank)

  //Nchan/Nthreads blocks, becomes Nchan threads for Nchan < Nthreads
    currChan = threadIdx.x + blockIdx.x * blockDim.x;
    int nspike = counter[0]-counter[1];

    while (currChan < Nchan) {
        for (ind = 0; ind < nspike; ++ind){
            currInd = idx[ind] + counter[1];
            const float *U0 = &U[currChan + Nchan*id[currInd]];
            int woff = nt0*id[currInd];
            int idataraw0 = st[currInd] + NT*currChan;
            for (int timeInd = 0; timeInd < nt0; ++timeInd){
                const float *W0 = &W[woff + timeInd];
                float X = 0.0f;
                for (k=0;k<Nrank;k++){
                    int nfk = Nfilt*k;
                    X += W0[nt0*nfk] * U0[Nchan*nfk];
                }
                dataraw[idataraw0 + timeInd] -= x[currInd] * X;
            }
        }
        currChan += blockDim.x * gridDim.x;
    }
}



//////////////////////////////////////////////////////////////////////////////////////////
__global__ void average_snips(const double *Params, const int *st, const unsigned int *idx,
        const int *id, const float *x, const float *y,  const int *counter, const float *dataraw,
        const float *W, const float *U, double *WU, int *nsp,
        const float *mu, const float *z){

  //threadIndex.x = 0-nt0-1
  //threadIndex.y = 0-15
  double X, xsum;
  float  Th;
  int nt0, tidx, tidy, bid, NT, Nchan,k, Nrank, Nfilt;
  int currInd, ind;


  NT        = (int) Params[0];
  Nfilt    	= (int) Params[1];
  nt0       = (int) Params[4];
  Nrank     = (int) Params[6];
  Nchan     = (int) Params[9];

  tidx 		= threadIdx.x;
  bid 		= blockIdx.x;

  //Th = 10.f;
  Th 		= (float) Params[15];

  // we need wPCA projections in here, and then to decide based on total

  // idx is the time sort order of the spikes; the original order is a function
  // of when threads complete in mexGetSpikes. Compilation of the sums for WU, sig, and dnextbest
  // in a fixed order makes the calculation deterministic.

  for(ind=0; ind<counter[0]; ind++) {
      currInd = idx[ind];
      // only do this if the spike is "GOOD"
      if (x[currInd]>Th){
          if (id[currInd]==bid){
              if (tidx==0 &&  threadIdx.y==0)
                  nsp[bid]++;

              tidy 		= threadIdx.y;
              while (tidy<Nchan){
                  X = 0.0f;
                  for (k=0;k<Nrank;k++)
                      X += W[tidx + bid* nt0 + nt0*Nfilt*k] *
                              U[tidy + bid * Nchan + Nchan*Nfilt*k];

                  xsum = dataraw[st[currInd]+tidx + NT * tidy] + y[currInd] * X;

                  //WU[tidx+tidy*nt0 + nt0*Nchan * bid] *= p[bid];
                  WU[tidx+tidy*nt0 + nt0*Nchan * bid] += (double) xsum;

                  tidy+=blockDim.y;

              }        //end of while loop over channels
          }               //end of if block for id == bid
      }
  }                  //end of for loop over spike indicies
}                      //end of function

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	computePCfeatures(const double *Params, const int *counter,
        const float *dataraw,  const int *st, const int *id, const float *x,
        const float *W, const float *U, const float *mu, const int *iW, const int *iC,
        const float *wPCA, float *featPC){

  volatile __shared__ float  sPCA[81 * NrankMax], sW[81 * NrankMax], sU[NchanMax * NrankMax];
  volatile __shared__ int iU[NchanMax];

  float X = 0.0f, Y = 0.0f;
  int bid, nt0, t, tidx, tidy, k, NT, ind, Nchan, NchanU, Nfilt, Nrank;

  NT        = (int) Params[0];
  nt0       = (int) Params[4];
  Nchan     = (int) Params[9];
  Nfilt    	= (int) Params[1];
  Nrank     = (int) Params[6];
  NchanU    = (int) Params[10];

  tidx 		= threadIdx.x;
  tidy 		= threadIdx.y;
  bid       = blockIdx.x;

  if (tidy==0)
      iU[tidx] = iC[tidx + NchanU * iW[bid]];
  __syncthreads();

  sU[tidx + tidy*NchanU]= U[iU[tidx] + Nchan * bid + Nchan * Nfilt * tidy];

  while (tidx<nt0){
     sW[tidx + tidy*nt0]  = W[tidx + bid*nt0 + Nfilt * nt0 * tidy];
      sPCA[tidx + tidy*nt0]  = wPCA[tidx + nt0 * tidy];
      tidx += blockDim.x;
  }

  tidx 		= threadIdx.x;
  __syncthreads();

//   first, compute wPCA projections of the filter
  Y = 0.0f;
  for (k =0; k<Nrank; k++){
      X = 0.0f;
      for (t=0;t<nt0;t++)
          X += sW[t + k*nt0] * sPCA[t + tidy * nt0];
      Y += X * sU[tidx + k*NchanU];
  }

  //now for each matching spike, compute the features
  for(ind=0; ind<counter[0];ind++)
      if (id[ind]==bid){
          X = Y * x[ind]; // - mu[bid]);
          for (t=0;t<nt0; t++)
              X  += dataraw[st[ind] + t + NT * iU[tidx]] * sPCA[t + nt0*tidy];
          featPC[tidx + tidy*NchanU + ind * NchanU*Nrank] = X;
      }
}

//////////////////////////////////////////////////////////////////////////////////////////
// This function is not called. 
__global__ void	addback_spikes(const double *Params,  const int *st, 
        const int *id, const float *x, const int *count, float *dataraw, 
        const float *W, const float *U, const int iter, const float *spkscore){

  float X, ThS;
  int nt0, tidx, tidy, k, NT, ind, Nchan, Nfilt, Nrank;

  NT        = (int) Params[0];
  nt0       = (int) Params[4];
  Nchan     = (int) Params[9];
  Nfilt    	= (int) Params[1];
  Nrank     = (int) Params[6];
  ThS       = (float) Params[11];

  tidx 		= threadIdx.x;
  ind       = count[iter]+blockIdx.x;

  while(ind<count[iter+1]){
      if (spkscore[ind]>ThS){

          tidy = threadIdx.y;
          // only do this if the spike is "BAD"
          while (tidy<Nchan){
              X = 0.0f;
              for (k=0;k<Nrank;k++)
                  X += W[tidx + id[ind]* nt0 + nt0*Nfilt*k] *
                          U[tidy + id[ind] * Nchan + Nchan*Nfilt*k];
              X = x[ind]*X;
              atomicAdd(&dataraw[tidx + st[ind] + NT * tidy], X);      
              tidy += blockDim.y;
          }
      }
      ind += gridDim.x;
  }
}

// create gpu array of starting index values, 0..nitimes-1
// call with no threads, i.e. <<1, 1>>
__global__ void set_idx( unsigned int *idx, const unsigned int nitems ) {
    for( int i = 0; i < nitems; ++ i ) {
        idx[i] = i;
    }
}
