const int  Nthreads = 1024,  NrankMax = 6, maxFR = 10000, nt0max=81, NchanMax = 17, nsizes = 5;


//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	Conv1D(const double *Params, const float *data, const float *W, float *conv_sig){
    volatile __shared__ float  sW[81*NrankMax], sdata[(Nthreads+81)];
    float y;
    int tid, tid0, bid, i, nid, Nrank, NT, nt0,  Nchan;

    tid 		= threadIdx.x;
    bid 		= blockIdx.x;

    NT        = (int) Params[0];
    Nchan     = (int) Params[1];
    nt0       = (int) Params[2];
    Nrank     = (int) Params[4];

    if(tid<nt0*Nrank)
        sW[tid]= W[tid];
    __syncthreads();

    tid0 = 0;
    while (tid0<NT-Nthreads-nt0+1){
        if (tid<nt0)
            sdata[tid] = data[tid0 + tid + NT*bid];
        sdata[tid + nt0] = data[nt0+tid0 + tid+ NT*bid];
        __syncthreads();

        for(nid=0;nid<Nrank;nid++){
            y = 0.0f;
            #pragma unroll 4
            for(i=0;i<nt0;i++)
                y    += sW[i + nid*nt0] * sdata[i+tid];
            conv_sig[tid0  + tid + NT*bid + nid * NT * Nchan]   = y;
        }
        tid0+=Nthreads;
        __syncthreads();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void  sumChannels(const double *Params, const float *data,
	float *datasum, int *kkmax, const int *iC2, const float *dist, const float *v2){

  int tid, tid0,t,k, kmax, bidx, bidy, NT, Nchan, NchanNear,j,iChan, Nsum, Nrank;
  float  Cmax, C0;
  float a[nsizes], d2;
  float  sigma;
  volatile __shared__ float  sA[nsizes * 20];


  tid 		= threadIdx.x;
  bidx 		= blockIdx.x;
  bidy 		= blockIdx.y;
  NT 		= (int) Params[0];
  Nchan     = (int) Params[1];
  NchanNear = (int) Params[3];
  Nrank     = (int) Params[4];
  Nsum      = (int) Params[3];
  sigma = (float) Params[9];

  if (tid<nsizes*NchanNear){
      d2 = dist[tid/nsizes + NchanNear * bidy];
      k = tid%nsizes;
      sA[tid] = expf( - (d2 * d2)/((1+k)*(1+k)*sigma*sigma));
  }
  __syncthreads();

  tid0 = tid + bidx * blockDim.x;
  while (tid0<NT){
      Cmax = 0.0f;
      kmax = 0;

      for (t=0;t<Nrank;t++){
          for(k=0; k<nsizes; k++)
              a[k] = 0.;

          for(j=0; j<Nsum; j++){
              iChan = iC2[j + NchanNear * bidy];
              for(k=0; k<nsizes; k++)
                  a[k]  += sA[k + nsizes * j] *
                        data[tid0 + NT * iChan + t * NT * Nchan];
          }
          for(k=0; k<nsizes; k++){
              a[k] = max(a[k], 0.);
              if (a[k]*a[k] / v2[k + nsizes*bidy] > Cmax){
                  Cmax = a[k]*a[k]/v2[k + nsizes*bidy];
                  kmax = t + k*Nrank;
               }
          }
      }
      datasum[tid0 + NT * bidy] = Cmax;
      kkmax[tid0 + NT * bidy]   = kmax;

      tid0 += blockDim.x * gridDim.x;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void	max1D(const double *Params, const float *data, float *conv_sig){

    volatile __shared__ float  sdata[Nthreads+81];
    float y, spkTh;
    int tid, tid0, bid, i, NT, nt0, nt0min;

    NT 		= (int) Params[0];
    nt0       = (int) Params[2];
    nt0min    = (int) Params[5];
    spkTh    = (float) Params[6];

    tid 		= threadIdx.x;
    bid 		= blockIdx.x;

    tid0 = 0;
    while (tid0<NT-Nthreads-nt0+1){
        if (tid<nt0)
            sdata[tid]   = data[tid0 + tid + NT*bid];
        sdata[tid + nt0] = data[nt0+tid0 + tid+ NT*bid];
        __syncthreads();

        y = 0.0f;
        #pragma unroll 4
        for(i=0;i<2*nt0min;i++)
            y    = max(y, sdata[tid+i]);

        if (y>spkTh*spkTh)
            conv_sig[tid0 + 1*(nt0min) + tid + NT*bid]   = y;

        tid0+=Nthreads;
        __syncthreads();
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
__global__ void  maxChannels(const double *Params, const float *dataraw, const float *data,
	const int *iC,  const int *iC2, const float *dist2, const int *kkmax,
        const float *dfilt, int *st, int *counter, float *cF){

  int nt0, indx, tid, tid0, i, bid, NT, j,iChan, nt0min, Nrank, kfilt;
  int Nchan, NchanNear, NchanUp, NchanNearUp, bidy ;
  double Cf, d;
  float spkTh, d2;
  bool flag;

  NT 		= (int) Params[0];
  Nchan     = (int) Params[1];
  NchanNear = (int) Params[3];
  NchanUp     = (int) Params[7];
  NchanNearUp = (int) Params[8];
  nt0       = (int) Params[2];
  nt0min    = (int) Params[5];
  spkTh    = (float) Params[6];
  Nrank     = (int) Params[4];

  tid 		= threadIdx.x;
  bid 		= blockIdx.x;
  bidy = blockIdx.y;

  tid0 = tid + bid * blockDim.x;
  while (tid0<NT-nt0-nt0min){
      i = bidy;
      Cf    = (double) data[tid0 + NT * i];
      flag = true;
      for(j=1; j<NchanNearUp; j++){
          if (dist2[j + NchanNearUp * i] < 100.){
              iChan = iC2[j+ NchanNearUp * i];
              if (data[tid0 + NT * iChan] > Cf){
                  flag = false;
                  break;
              }
          }
      }

      if (flag){
          if (Cf>spkTh*spkTh){
              d = (double) dataraw[tid0+0 * (nt0min-1) + NT*i]; //
              if (d > Cf-1e-6){
                  // this is a hit, atomicAdd and return spikes
                  indx = atomicAdd(&counter[0], 1);
                  if (indx<maxFR){
                      st[0+4*indx] = tid0;
                      st[1+4*indx] = i;
                      st[2+4*indx] = sqrt(d);
                      st[3+4*indx] = kkmax[tid0+0*(nt0min-1) + NT*i];
                      kfilt = st[3+4*indx]%Nrank;
                      for(j=0; j<NchanNear; j++){
                          iChan = iC[j+ NchanNear * i];
                          cF[j + NchanNear * indx] = dfilt[tid0+0*(nt0min-1) + NT * iChan + kfilt * Nchan*NT];
                      }
                  }
              }
          }
      }

      tid0 += blockDim.x * gridDim.x;
  }
}
