    float *res_count = output + bnIdx * outputElem;
    int count = (int)atomicAdd(res_count, 1);
