//
// Copyright (c) 2018 Kohei Nagasawa
// Read LICENSE.md for license condition of this software
//

// Utility functions for cuda
namespace CudaUtils
{
    // Allocate array on device
    void allocateArray(void** ptr, const size_t size);

    // Free array stored on device
    void freeArray(void** ptr);

    // Copy data from host to device
    void cpyHostToDevice(void* dataDevice, void* dataHost, const size_t size);

    // Copy data from device to host
    void cpyDeviceToHost(void* dataHost, void* dataDevice, const size_t size);

    // Copy data in device
    void cpyDeviceToDevice(void* dataDevice1, void* dataDevice2, const size_t size);

    // Compute the number of blocks and threads
    void computeGridSize(unsigned int numComputeUnits, unsigned int blockSize, unsigned int&, unsigned int&);
}