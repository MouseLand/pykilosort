const int nblock = 32;
//////////////////////////////////////////////////////////////////////////////////////////

__global__ void	crossFilter(const float *W1, const float *W2, const float *UtU,
        float *WtW, int Nfilt, int nt0){

  float x;
  int tidx, tidy , bidx, bidy, i, t, tid1, tid2;

  tidx 		= threadIdx.x;
  tidy 		= threadIdx.y;
  bidx 		= blockIdx.x;
  bidy 		= blockIdx.y;

  tid1 = tidx + bidx*nblock;
  tid2 = tidy + bidy*nblock;

  if (tid2<Nfilt && tid1<Nfilt){
      for(i=0;i<2*nt0-1;i++){
          x = 0.0f;
          if(i<nt0)
              for(t=0;t<i+1;t++)
                  x += W1[t + nt0 * tid1] * W2[t + (nt0-i-1) + nt0 * tid2];
          else
              for(t=i-nt0+1;t<nt0;t++)
                  x += W1[t + nt0 * tid1] * W2[t + (nt0-i-1) + nt0 * tid2];

          WtW[tid1 + tid2*Nfilt +  i*Nfilt*Nfilt] =
                  x * UtU[tid1 + tid2*Nfilt];
      }
  }
}
