//
// Copyright (c) 2018 Kohei Nagasawa
// Read LICENSE.md for license condition of this software
//

#pragma warning(disable : 4819) // Disable warning for invalid character
#include "cudaUtils.cuh"
#include <helper_math.h>

inline unsigned int iDivUp(unsigned int a, unsigned int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

namespace CudaUtils
{
    // Allocate array on device
    void allocateArray(void** ptr, const size_t size)
    {
        cudaThreadSynchronize();
        cudaMalloc(ptr, size);
    }

    // Free array stored on device
    void freeArray(void** ptr)
    {
        cudaFree(ptr);
        *ptr = nullptr;
    }

    // Copy data from host to device
    void cpyHostToDevice(void *dataDevice, void *dataHost, const size_t size)
    {
        cudaMemcpy(dataDevice, dataHost, size, cudaMemcpyHostToDevice);
    }

    // Copy data from device to host
    void cpyDeviceToHost(void *dataHost, void *dataDevice, const size_t size)
    {
        cudaMemcpy(dataHost, dataDevice, size, cudaMemcpyDeviceToHost);
    }

    // Copy data in device
    void cpyDeviceToDevice(void *dataDevice1, void *dataDevice2, const size_t size)
    {
        cudaMemcpy(dataDevice1, dataDevice2, size, cudaMemcpyDeviceToDevice);
    }

    // Compute the number of blocks and threads
    void computeGridSize(unsigned int n, unsigned int blockSize, unsigned int& numBlocksOut, unsigned int& numThreadsOut)
    {
        numThreadsOut = min(blockSize, n);
        numBlocksOut = iDivUp(n, numThreadsOut);
    }
}