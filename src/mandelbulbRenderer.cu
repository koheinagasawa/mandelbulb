//
// Copyright (c) 2018 Kohei Nagasawa
// Read LICENSE.md for license condition of this software
//

#include "mandelbulbRenderer.cuh"
#include "cudaUtils.cuh"

#ifdef ENABLE_DOUBLE_PRECISION
#include "double3.h"
#endif

#include <cmath>
#include <stdio.h>
#include <time.h>

#define FLOAT_MAX 3.402823466e+38F

#define ANALYTICAL_DISTANCE_ESTIMATION

using namespace CudaUtils;

// Constants
static constexpr float s_sphereRadiusSquared = 1.22f;
static constexpr float s_mandelEvalThreshold = 2.f;
static constexpr float s_rayAcceleration = 0.000002f;
static constexpr float s_deltaPower = 2000000.0f;
static constexpr float s_maxDepth = 1.5f;
static constexpr float s_minDepth = 0.0002f;
static constexpr int s_numBinaryIterations = 6;
static constexpr int s_numIterationsOnHost = 20;

// Tables for Morton curve
static constexpr unsigned int s_mortonMasksHost[] = { 0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF };
__device__ const unsigned int s_mortonMasks[]     = { 0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF };

// Texture buffers
texture<float, 1, cudaReadModeElementType> s_positionTexture;
texture<float, 1, cudaReadModeElementType> s_depthTexture;
texture<float, 1, cudaReadModeElementType> s_deltaDepthTexture;
texture<float, 1, cudaReadModeElementType> s_depthDiffTexture;
texture<int,   1, cudaReadModeElementType> s_iterationTexture;

#define ENCODE_MORTON_CURVE(MASK) \
    unsigned int x = xPos; \
    unsigned int y = yPos; \
    x = (x | (x << 8)) & MASK[3]; \
    x = (x | (x << 4)) & MASK[2]; \
    x = (x | (x << 2)) & MASK[1]; \
    x = (x | (x << 1)) & MASK[0]; \
    y = (y | (y << 8)) & MASK[3]; \
    y = (y | (y << 4)) & MASK[2]; \
    y = (y | (y << 2)) & MASK[1]; \
    y = (y | (y << 1)) & MASK[0]; \
    const unsigned int result = x | (y << 1); \
    return result

// Encode Morton curve on CPU
unsigned int encodeMortonCurveHost(unsigned short xPos, unsigned short yPos)
{
    ENCODE_MORTON_CURVE(s_mortonMasksHost);
}

// Evaluate Mandelbulb at the given position on host CPU
template <int N>
bool evalMandelbulbOnHost(const float3& pos, int numIterations, int& iterationLeft)
{
    float3 hc = pos;
    float r;

    // Evaluate mandelbulb at the point with the given iteration count
    for (int i = 0; i < numIterations; i++)
    {
        if (hc.x == 0)
        {
            iterationLeft = numIterations - i;
            break;
        }

        r = sqrt(hc.x * hc.x + hc.y * hc.y + hc.z * hc.z);

        if (r > s_mandelEvalThreshold)
        {
            // r is diverged, which means we will never hit the surface.
            // Abort iteration.
            iterationLeft = numIterations - i;
        }

        if (hc.x != 0)
        {
            float phi = atan(hc.y / hc.x);
            float theta = acos(hc.z / r);

            r = pow(r, N);

            theta = N * theta;
            phi = N * phi;
            const float sinth = sin(theta) * r;

            hc.x = sinth * cos(phi);
            hc.y = sinth * sin(phi);
            hc.z = cos(theta) * r;
        }

        hc = hc + pos;
    }

    // We didn't diverged withint the iteration count.
    // Then this point is considered as surface of mandelbulb.
    iterationLeft = 0;
    return true;
}

// Perform ray marching from the center of view on CPU
void MandelbulbRenderer::rayMarchOnHost(float initialDeltaDepth)
{
    printf("----- Begin Ray Marching -----\n");

    const float3& direction = m_cameraForward;
    float3 originalPosition = m_cameraPosition;
    {
        const float cameraDistanceSquared = dot(originalPosition, originalPosition);

        if (cameraDistanceSquared >= s_sphereRadiusSquared)
        {
            // Camera is outside of the radius.
            // Ray should start from sphere's surface of the radius
            originalPosition = originalPosition + direction * sqrt(cameraDistanceSquared - s_sphereRadiusSquared);
        }
    }

    float3 pos = originalPosition;
    float deltaDepth = initialDeltaDepth > 0 ? initialDeltaDepth : m_initialDeltaStep;

    // Get iteration count
    const int numIterations = 10;
    while (1)
    {
        // Check if we are still inside the radius
        if (dot(pos, pos) > s_sphereRadiusSquared)
        {
            break;
        }

        // Evaluate mandelbulb
        bool result;
        int iterationLeft;
        {
            // Evaluate Mandelbulb
            result = evalMandelbulbOnHost<8>(pos, numIterations, iterationLeft);
        }

        // Calculate the current depth
        float depth = dot(pos - originalPosition, direction);
        printf("%f,%d\n", depth, numIterations - iterationLeft);

        if (result)
        {
            // We got a hit!
            printf("----- Hit : End of Ray Marching -----\n");
            return;
        }

        // Update depth and position
        pos = pos + direction * deltaDepth;
    }

    printf("----- No Hit : End of Ray Marching -----\n");
    return;
}

namespace MandelbulbCudaKernel
{

// Get pixel index
__device__
unsigned int getPixelIndex()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

// Encode pixel coordenates into morton curve index
__device__
unsigned int encodeMortonCurve(unsigned short xPos, unsigned short yPos)
{
    ENCODE_MORTON_CURVE(s_mortonMasks);
}

// Decode morton curve index into pixel coordinates
__device__
void decodeMortonCurve(unsigned int morton, unsigned short& xPos, unsigned short& yPos)
{
    unsigned int x = morton & s_mortonMasks[0];
    unsigned int y = (morton & (s_mortonMasks[0] << 1)) >> 1;

    x = (x | x >> 1) & s_mortonMasks[1];
    x = (x | x >> 2) & s_mortonMasks[2];
    x = (x | x >> 4) & s_mortonMasks[3];
    x = (x | x >> 8) & s_mortonMasks[4];

    y = (y | y >> 1) & s_mortonMasks[1];
    y = (y | y >> 2) & s_mortonMasks[2];
    y = (y | y >> 4) & s_mortonMasks[3];
    y = (y | y >> 8) & s_mortonMasks[4];

    xPos = x;
    yPos = y;
}

// Get index of pixel from its coordinate
__device__
unsigned int getIndex(int x, int y, int width, bool mortonCurve)
{
    if (mortonCurve)
    {
        return encodeMortonCurve((unsigned short)x, (unsigned short)y);
    }
    else
    {
        return x + y * width;
    }
}

// Get index of neighbor pixel
__device__
unsigned int getNeighborIndex(unsigned int index, int xOffset, int yOffset, int width, bool mortonCurve)
{
    if (mortonCurve)
    {
        unsigned short x, y;
        decodeMortonCurve(index, x, y);
        return encodeMortonCurve(x + (unsigned short)xOffset, y + (unsigned short)yOffset);
    }
    else
    {
        return index + xOffset + width * yOffset;
    }
}

// Set coordinates of each pixcel
__global__
void setTexcoords(unsigned int numPixcels, int width, int height, bool mortonCurve, int2* texcoords)
{
    const unsigned int id = getPixelIndex();
    if (id < numPixcels)
    {
        int2& cord = texcoords[id];

        if (mortonCurve)
        {
            unsigned short x, y;
            decodeMortonCurve(id, x, y);
            cord.x = x;
            cord.y = y;
        }
        else
        {
            const int widthIndex = id % width;
            const int heightIndex = (id - widthIndex) / width;
            cord.x = widthIndex;
            cord.y = heightIndex;
        }
    }
}

// Rotates a vector by angle around axis
__device__
float3 rotate(const float3& vector, const float3& axis, const float angle)
{
    float3 vectorOut;
    const float c = cos(angle);
    const float s = sin(angle);
    const float cosx = (1 - c) * axis.x;
    const float cosy = (1 - c) * axis.y;
    const float cosz = (1 - c) * axis.z;
    const float sinx = s * axis.x;
    const float siny = s * axis.y;
    const float sinz = s * axis.z;
    const float cosxy = cosx * axis.y;
    const float cosxz = cosx * axis.z;
    const float cosyz = cosy * axis.z;

    vectorOut.x = (c + cosx * axis.x) * vector.x
                + (cosxy - sinz) * vector.y
                + (cosxz + siny) * vector.z;
    vectorOut.y = (cosxy + sinz) * vector.x
                + (c + cosy * axis.y) * vector.y
                + (cosyz - sinx) * vector.z;
    vectorOut.z = (cosxz - siny) * vector.x
                + (cosyz + sinx) * vector.y
                + (c + cosz * axis.z) * vector.z;

    return vectorOut;
}

// Set ray marching directions for each pixel
__global__
void setPixelDirection(
    const unsigned int numPixels,
    const int halfWidth,
    const int halfHeight,
    const int2* texcoords,
    const float3 forwardDir,
    const float3 upDir,
    const float3 sideDir,
    const float dagl,
    float3* pixelDirs)
{
    const unsigned int id = getPixelIndex();
    if (id < numPixels)
    {
        const int2& cord = texcoords[id];
        const float xAngle = (cord.x - halfWidth) * dagl;
        const float yAngle = (cord.y - halfHeight) * dagl;
        float3& pixelDir = pixelDirs[id];
        pixelDir = rotate(forwardDir, upDir, xAngle);
        pixelDir = rotate(pixelDir, sideDir, yAngle);
        pixelDir = normalize(pixelDir);
    }
}

// Set ray marching directions for each pixel
__global__
void setPixelDirectionLow(
    const unsigned int numPixels,
    const int halfWidth,
    const int halfHeight,
    const int2* texcoords,
    const float3 forwardDir,
    const float3 upDir,
    const float3 sideDir,
    const float dagl,
    float3* pixelDirs)
{
    const unsigned int id = getPixelIndex();
    if (id < numPixels)
    {
        const int2& cord = texcoords[id];
        float xAngle;
        if (cord.y % 2 == 0)
        {
            xAngle = (2 * cord.x - halfWidth) * dagl;
        }
        else
        {
            xAngle = (2 * cord.x + 1 - halfWidth) * dagl;
        }
        const float yAngle = (cord.y - halfHeight) * dagl;
        float3& pixelDir = pixelDirs[id];
        pixelDir = rotate(forwardDir, upDir, xAngle);
        pixelDir = rotate(pixelDir, sideDir, yAngle);
        pixelDir = normalize(pixelDir);
    }
}

// Actual evaluation of mandelbulb at given position
// Depth is used to determined the number of iterations
// Returns 1 if we hit mandelbulb or negative value if we didn't
// The smaller the returned value is, the sooner iteration ended
// which means the further the point is from the surface
template <int N>
__device__
bool evalMandelbulb(const float3& pos, const int numIterations, int& iterationLeft, float& potential, float& dr)
{
    float3 hc = pos;
    float r = 0.0f;
    dr = 1.0f;

    // Evaluate mandelbulb at the point with the given iteration count
    for (int i = 0; i < numIterations; i++)
    {
        if (hc.x == 0)
        {
            iterationLeft = numIterations - i;

            potential = log(r) / pow((float)N, float(i));
            dr = 0.5f * log(r) * r / dr;

            return false;
        }

        r = sqrt(dot(hc, hc));

        if (r > s_mandelEvalThreshold)
        {
            // r is diverged, which means we will never hit the surface.
            // Abort iteration.

            iterationLeft = numIterations - i;
            potential = log(r) / pow((float)N, float(i));
            dr = 0.5f * log(r) * r / dr;

            return false;
        }

        if (hc.x != 0)
        {
            float phi = atan(hc.y / hc.x);
            float theta = acos(hc.z / r);

            r = pow(r, N);

            theta = N * theta;
            phi = N * phi;
            const float sinth = sin(theta) * r;

            hc.x = sinth * cos(phi);
            hc.y = sinth * sin(phi);
            hc.z = cos(theta) * r;
        }

        hc += pos;

        dr = pow(r, N - 1) * (float)N * dr + 1.0f;
    }

    // We didn't diverged withint the iteration count.
    // Then this point is considered as surface of mandelbulb.

    iterationLeft = 0;
    potential = 0;
    return true;
}

template <int N>
__device__
float potential(int numIteration, const float3& pos)
{
    float3 hc = pos;
    float r;
    for (int i = 0; i < numIteration; ++i)
    {
        r = sqrt(dot(hc, hc));

        if (r > s_mandelEvalThreshold)
        {
            return log(r) / pow((float)N, float(i));
        }

        float theta = acos(hc.z / r);
        float phi = atan(hc.y / hc.x);
        float zr = pow(r, (float)N);
        theta = theta*(float)N;
        phi = phi*(float)N;

        const float sinth = sin(theta) * zr;

        hc.x = sinth * cos(phi);
        hc.y = sinth * sin(phi);
        hc.z = cos(theta) * zr;

        hc = pos + hc;
    }

    return 0;
}

// Calculate the number of iterations used for Mandelbulb evaluation
__device__
int getNumIterations(float depth, float iterationAccelerator, int minimumIterations)
{
    return (int)(-log(pow(depth, iterationAccelerator)) / log(50.f)) + minimumIterations;
}

// Evaluate Mandelbulb by ray marching
template <int N>
__device__
bool rayMarchWithAcceleration(
    float3 pos,
    const float3& direction,
    float iterationAccelerator,
    int minimumIterations,
    int fixedIteration,
    bool distanceEstimation,
    int& pixelIteration,
    int& numSteps,
    float& depth,
    float& deltaDepth)
{
    float rayAccelerator = 0;
    float pot, dr;
    int currentIteration = -1;
    int iterationLeft;
    bool result;

    numSteps = 0;
    while (1)
    {
        ++numSteps;

        // Check if we are still inside the radius
        if (dot(pos, pos) > s_sphereRadiusSquared)
        {
            break;
        }

        // Evaluate mandelbulb
        {
            currentIteration = fixedIteration > 0 ? fixedIteration : getNumIterations(depth, iterationAccelerator, minimumIterations);

            // Evaluate Mandelbulb
            result = evalMandelbulb<N>(pos, currentIteration, iterationLeft, pot, dr);

            // Update iteration count of this pixel
            pixelIteration = currentIteration;

            // [todo] Review and validate this code
            //if (!result && iterationLeft > 1)
            //{
            //    // Iteration count is too big. Make it smaller
            //    minimumIterations = currentIteration - iterationLeft + 1;
            //}
        }

        if (result)
        {
            // We got a hit!
            return true;
        }

        // Update depth and position
        float delta = deltaDepth;
        float acceleration = s_rayAcceleration;
        if (distanceEstimation)
        {
#ifndef ANALYTICAL_DISTANCE_ESTIMATION // Approximate derivative by gradient
            const float eps = 0.01f * deltaDepth;
            float3 posx = pos; posx.x += eps;
            float3 posy = pos; posy.y += eps;
            float3 posz = pos; posz.z += eps;
            float3 grad;
            grad.x = potential<N>(currentIteration, posx);
            grad.y = potential<N>(currentIteration, posy);
            grad.z = potential<N>(currentIteration, posz);
            grad = (grad - pot) / eps;
            float newDelta = max(delta, (0.5f / exp(pot))*sinh(pot) / sqrt(dot(grad, grad)));
#else // Analytical solution
            float newDelta = max(delta, dr);
#endif

            acceleration *= newDelta / delta;
            delta = newDelta;
        }

        depth += delta;
        pos = pos + direction * delta;

        // Accelerate the ray gradually while it gets far from the camera
        rayAccelerator += acceleration;
        deltaDepth *= pow(s_deltaPower, rayAccelerator + iterationLeft * acceleration); // [todo] Investigate this calculation
    }

    return false;
}

// Perform ray marching against mandelbulb
// and store depth to mandelbulb and delta depth of ray marching for each pixel
template <int N>
__global__
void raymarchMandelbulb_kernel(
    const unsigned int numPixels,
    const float3* pixelDirs,
    const float3 cameraPosition,
    float deltaDepth,
    float iterationAccelerator,
    int minimumIterations,
    bool distanceEstimation,
    int* pixelIterations,
    int* numRayMarchSteps,
    float* pixelDepths,
    float* pixelDeltaDepths)
{
    const unsigned int id = getPixelIndex();
    if (id >= numPixels)
    {
        return;
    }

    // Reset iteration count of mandelbulb evaluation
    pixelIterations[id] = 0;

    float3 pos;
    const float3& direction = pixelDirs[id];

    // First, check if we need to evaluate mandelbulb from the first place
    bool shouldEval = false;
    {
        const float cameraDistanceSquared = dot(cameraPosition, cameraPosition);
        const float projection = dot(cameraPosition, direction);
        if (projection > 0)
        {
            // Camera is facing the opposite way of the origin
            // We don't need to evaluate at all if the camera is outside of the radius

            if (cameraDistanceSquared < s_sphereRadiusSquared)
            {
                // If the camera is inside the radius, ray should start from there
                pos = cameraPosition + deltaDepth * direction;
                shouldEval = true;
            }
        }
        else
        {
            // Camera is facing to mandelbulb
            // Now check if the direction of ray can possibly hit mandelbulb
            const float dist = cameraDistanceSquared - projection * projection;
            if (dist < s_sphereRadiusSquared)
            {
                // Ray can hit mandelbulb
                // Now determine the start position of the ray

                if (cameraDistanceSquared < s_sphereRadiusSquared)
                {
                    // If the camera is inside the radius, ray should start from there
                    pos = cameraPosition + deltaDepth * direction;
                }
                else
                {
                    // Camera is outside of the radius.
                    // Ray should start from sphere's surface of the radius
                    float d1 = -projection;
                    float d2 = sqrt(s_sphereRadiusSquared - dist);
                    pos = cameraPosition + direction * (d1 - d2 + deltaDepth);
                }

                shouldEval = true;
            }
        }
    }

    // Secondly, evaluate mandelbulb if needed
    if (shouldEval)
    {
        // Calculate the current depth
        float depth = dot(pos - cameraPosition, direction);

        // Perform ray marching
        bool hit = rayMarchWithAcceleration<N>(
            pos, 
            direction, 
            iterationAccelerator,
            minimumIterations,
            -1,
            distanceEstimation,
            pixelIterations[id],
            numRayMarchSteps[id],
            depth, 
            deltaDepth);

        if (hit)
        {
            // We hit mandelbulb.
            // Store depth and delta depth of the hit
            pixelDepths[id] = depth;
            pixelDeltaDepths[id] = deltaDepth;
            return;
        }
    }

    // No hit
    pixelDepths[id] = FLOAT_MAX;
    pixelDeltaDepths[id] = 1.0f;
    pixelIterations[id] = 0;

    return;
}

// Perform bisection search to get surface position with higher precision
template <int N>
__global__
void binaryPartitionSearch(
    const unsigned int numPixels,
    const float3* pixelDirs,
    const float* pixelDeltaDepths,
    const float3 cameraPosition,
    const int* pixelIterations,
    float3* pixelPositions,
    float* pixelDepths)
{
    const unsigned int id = getPixelIndex();
    if (id >= numPixels)
    {
        return;
    }

    float depth = pixelDepths[id];

    if (depth != FLOAT_MAX)
    {
        // We have hit mandelbulb at this pixel

        const float3& dir = pixelDirs[id];
        const float deltaDepth = pixelDeltaDepths[id];
        float currentDelta = deltaDepth;

        // Move backward one step
        float3 pos = cameraPosition + dir * (depth - deltaDepth);

        // Perform bi-section search
        for (int i = 0; i < s_numBinaryIterations; i++)
        {
            // Move half way
            currentDelta *= 0.5f;
            pos = pos + dir * currentDelta;
            depth += currentDelta;

            int numIterations = pixelIterations[id];
            int iterationLeft;
            float pot, grad;
            if (evalMandelbulb<N>(pos, numIterations, iterationLeft, pot, grad))
            {
                // We got a hit
                // Revert the last step and test again with smaller delta
                pos = pos - dir * currentDelta;
                depth -= currentDelta;
            }
        }

        // Store the result
        pixelDepths[id] = depth + currentDelta;
        pixelPositions[id] = pos + dir * currentDelta;
    }
    else
    {
        // There was no hit
        // Just set invalid position
        pixelPositions[id] = { FLOAT_MAX, FLOAT_MAX, FLOAT_MAX };
    }
}

// Drill down mandelbulb's surface by changing iteration count adaptively based on smoothness of the surface
// This kernel should be followed by compareDepthDiffs_kernel
template <int N>
__global__
void adaptiveIteration_kernel(
    const unsigned int numPixels,
    const float3* pixelDirs,
    const float3 cameraPosition,
    int minimumIterations,
    float iterationToDeltaStepRate,
    bool distanceEstimation,
    int* pixelIterations,
    float* pixelDepths,
    float* pixelDepthsTmp,
    float* pixelDeltaDepths,
    float* depthDiffs)
{
    const unsigned int id = getPixelIndex();
    if (id >= numPixels)
    {
        return;
    }

    if (pixelIterations[id] <= 0)
    {
        // This pixel is already done. Abort further drilling.
        return;
    }

    // Increment iteration count by one
    int currentIteration = pixelIterations[id] + 1;

    const float3& dir = pixelDirs[id];
    float deltaDepth = pixelDeltaDepths[id] * iterationToDeltaStepRate;
    float depth = pixelDepths[id];
    float3 pos = cameraPosition + dir * depth;
    bool hit = false;

    // Perform ray marching with the new iteration count but without ray acceleration
    while (1)
    {
        // Check if we are still inside the radius
        if (dot(pos, pos) > s_sphereRadiusSquared)
        {
            break;
        }

        // Evaluate mandelbulb
        int iterationLeft;
        float pot, dr;
        bool result = evalMandelbulb<N>(pos, currentIteration, iterationLeft, pot, dr);

        if (result)
        {
            // We hit the surface
            hit = true;
            break;
        }
        // [todo] Review and validate this code
        //
        //else if (iterationLeft > 1)
        //{
        //    // We didn't hit the surface and the iteration count is too big now.
        //    // This means the ray is now moving aways from the surface.
        //    // Reduce the iteration count and perform ray march with acceleration again.

        //    // Calculate new iteration count and update delta depth
        //    int numIterations = max(currentIteration - iterationLeft + 1, minimumIterations);
        //    deltaDepth *= float(iterationLeft - 1) / iterationToDeltaStepRate;

        //    // Perform ray marching
        //    int dummy;
        //    hit = rayMarchWithAcceleration<N>(
        //        pos,
        //        dir,
        //        0.f,
        //        0,
        //        numIterations,
        //        distanceEstimation,
        //        dummy,
        //        depth,
        //        deltaDepth
        //        );

        //    if (hit)
        //    {
        //        // We hit a new surface
        //        // Continue the loop from there with new parameters

        //        pos = cameraPosition + dir * depth;
        //        pixelDepths[id] = depth;
        //        pixelDeltaDepths[id] = deltaDepth;
        //        pixelIterations[id] = numIterations;

        //        currentIteration = numIterations + 1;
        //        deltaDepth *= iterationToDeltaStepRate;

        //        hit = false;
        //        isFirstStep = true;

        //        //continue;
        //    }

        //    // We didn't hit anything.
        //    // Just break the loop to make this pixel invalid.
        //    //break;
        //}

        // Update depth and position and evaluate mandelbulb again
        float delta = deltaDepth;
        if (distanceEstimation)
        {
#ifndef ANALYTICAL_DISTANCE_ESTIMATION // Approximate derivative by gradient
            const float eps = 0.01f * deltaDepth;
            float3 posx = pos; posx.x += eps;
            float3 posy = pos; posy.y += eps;
            float3 posz = pos; posz.z += eps;
            float3 grad;
            grad.x = potential<N>(currentIteration, posx);
            grad.y = potential<N>(currentIteration, posy);
            grad.z = potential<N>(currentIteration, posz);
            grad = (grad - pot) / eps;
            float newDelta = max(delta, (0.5f / exp(pot))*sinh(pot) / sqrt(dot(grad, grad)));
#else // Analytical solution
            float newDelta = max(delta, dr);
#endif

            delta = newDelta;
        }

        depth += delta;
        pos = pos + dir * delta;
    }

    if (hit)
    {
        // When we hit a new surface with incremented iteration count, set depth difference from the original and temporary depth
        depthDiffs[id] = (depth - pixelDepths[id]) / depth;
        pixelDepthsTmp[id] = depth;
    }
    else
    {
        // We didn't hit any surface with the new iteration
        // Just set invalid values
        pixelDepths[id] = FLOAT_MAX;
        depthDiffs[id] = FLOAT_MAX;
        pixelIterations[id] = 0;
    }
}

// Compare depth difference of nearby pixels and calculate screen space laplacian of it to determine
// if the surface is still smooth enough to drill down further
__global__
void compareDepthDiffs_kernel(
    const unsigned int numPixels,
    const int width,
    const int height,
    const float laplacianThreshold,
    const float* pixelDepthDiffs,
    const float* pixelDepthsTmp,
    const int2* texcoords,
    float iterationToDeltaStepRate,
    bool mortonCurve,
    bool upsampling,
    int* pixelIterations,
    float* pixelDepths,
    float* pixelDeltaDepths)
{
    const unsigned int id = getPixelIndex();
    if (id >= numPixels)
    {
        return;
    }

    const float depth = pixelDepths[id];

    if (depth == FLOAT_MAX)
    {
        return;
    }

    const float diff = tex1Dfetch(s_depthDiffTexture, id);
    const float depthPlusDiff = depth + diff * pixelDepthsTmp[id];

    const int2& cord = texcoords[id];
    const int x = cord.x;
    const int y = cord.y;

    // Threshold of difference of depth between two pixels to calculate second derivative of depth diff
    // [todo] investigate if we need this condiyion
    //const float threshold = FLOAT_MAX;
    //const float invDepth = 1.f / depthPlusDiff;

    // Gradient of depth difference at neighbor pixels
    float gradDiffX1, gradDiffX2;
    float gradDiffY1, gradDiffY2;

    const int neighborOffsetX = upsampling ? 1 : 2;
    const int neighborOffsetY = 2;

    // [todo] Laplacian should be calculated only based on neighbors with the same iteration count as this pixel.
    //        We are not checking iteration count of neighbors now but we should.

    unsigned int neighbor;
    float neighborDiff;
    // Left pixel
    if (x > 1)
    {
        neighbor = getNeighborIndex(id, -neighborOffsetX, 0, width, mortonCurve);
        //if (fabs(depth - tex1Dfetch(s_depthTexture, neighbor)) * invDepth < threshold)
        {
            neighborDiff = tex1Dfetch(s_depthDiffTexture, neighbor);
            gradDiffX1 = diff - neighborDiff;
        }
        //else
        //{
        //    gradDiffX1 = 0;
        //}
    }
    else
    {
        gradDiffX1 = 0;
    }

    // Right pixel
    if (x < width - neighborOffsetX)
    {
        neighbor = getNeighborIndex(id, neighborOffsetX, 0, width, mortonCurve);
        //if (fabs(depth - tex1Dfetch(s_depthTexture, neighbor)) * invDepth < threshold)
        {
            neighborDiff = tex1Dfetch(s_depthDiffTexture, neighbor);
            gradDiffX2 = neighborDiff - diff;
        }
        //else
        //{
        //    gradDiffX2 = 0;
        //}
    }
    else
    {
        gradDiffX2 = 0;
    }

    // Up pixel
    if (y > 1)
    {
        neighbor = getNeighborIndex(id, 0, -neighborOffsetY, width, mortonCurve);
        //if (fabs(depth - tex1Dfetch(s_depthTexture, neighbor)) * invDepth < threshold)
        {
            neighborDiff = tex1Dfetch(s_depthDiffTexture, neighbor);
            gradDiffY1 = diff - neighborDiff;
        }
        //else
        //{
        //    gradDiffY1 = 0;
        //}
    }
    else
    {
        gradDiffY1 = 0;
    }

    // Down pixel
    if (y < height - neighborOffsetY)
    {
        neighbor = getNeighborIndex(id, 0, neighborOffsetY, width, mortonCurve);
        //if (fabs(depth - tex1Dfetch(s_depthTexture, neighbor)) * invDepth < threshold)
        {
            neighborDiff = tex1Dfetch(s_depthDiffTexture, neighbor);
            gradDiffY2 = neighborDiff - diff;
        }
        //else
        //{
        //    gradDiffY2 = 0;
        //}
    }
    else
    {
        gradDiffY2 = 0;
    }

    // Calculate laplacian of depth difference
    const float laplacian = (gradDiffX2 - gradDiffX1) + (gradDiffY2 - gradDiffY1);

    if (fabs(laplacian) < laplacianThreshold)
    {
        // If laplacian is smaller than the threshold, it means the surface is smooth enough to drill down one more iteration
        pixelIterations[id] += 1;
        pixelDepths[id] = depthPlusDiff;
        pixelDeltaDepths[id] *= iterationToDeltaStepRate;
    }
    else
    {
        // Otherwise, negate the iteration count to indicate we cannot drill down this pixel any further
        pixelIterations[id] = -pixelIterations[id];
    }
}

// Finalize data after adaptive drilling down
__global__
void finalizeIteration_kernel(
    const unsigned int numPixels,
    //const int width,
    int* pixelIterations)
{
    const unsigned int id = getPixelIndex();
    if (id >= numPixels)
    {
        return;
    }

    // Set negative iteration count back to the original positive value
    if (pixelIterations[id] < 0)
    {
        pixelIterations[id] = -pixelIterations[id];
    }

    // Take average of neaby iteration counts
    // [todo] Investigate if we really need this process.
    //        It seems it could just increase artifacts.
    //
    //const int iteration = pixelIterations[id];
    //int count = 1;
    //int sum = iteration;
    //for (int i = -2; i <= 2; ++i)
    //{
    //    for (int j = -2; j <= 2; ++j)
    //    {
    //        //int neighbor = id + i + width * j;
    //        const int neighbor = getNeighborIndex(id, i, j, width);
    //        const int neighborIteration = tex1Dfetch(s_iterationTexture, neighbor);

    //        if (abs(iteration - neighborIteration) < 3)
    //        {
    //            ++count;
    //            sum += neighborIteration;
    //        }
    //    }
    //}
    //pixelIterations[id] = sum / count;
}

// Copy data from low resolution buffer to high resolution buffer
__global__
void copyFromLowToHigh(
    const unsigned int numPixels,
    int width,
    bool mortonCurve,
    const int2* texcoordsLow,
    const float* pixelDepthsLow,
    const float* pixelDeltaDepthsLow,
    const int* pixelIterationsLow,
    float* pixelDepths,
    float* pixelDeltaDepths,
    int* pixelIterations)
{
    const unsigned int id = getPixelIndex();
    if (id >= numPixels)
    {
        return;
    }

    // Calculate pixel index of high resolution buffers
    const int2& texcoord = texcoordsLow[id];
    const int x = texcoord.y % 2 == 0 ? texcoord.x * 2 : texcoord.x * 2 + 1;
    unsigned int highResId = getIndex(x, texcoord.y, width, mortonCurve);

    pixelDepths[highResId] = pixelDepthsLow[id];
    pixelDeltaDepths[highResId] = pixelDeltaDepthsLow[id];
    pixelIterations[highResId] = pixelIterations[id];
}

__device__
void getValuesFromClosestNeighbor(
    const unsigned int neighbor,
    float& closestDepth,
    float& closestDeltaDepth,
    int& closestIteration,
    int& maxIteration
)
{
    float depth = tex1Dfetch(s_depthTexture, neighbor);
    int iteration = tex1Dfetch(s_iterationTexture, neighbor);
    if (depth < closestDepth)
    {
        closestDepth = depth;
        closestIteration = iteration;
        closestDeltaDepth = tex1Dfetch(s_deltaDepthTexture, neighbor);
    }
    if (iteration > maxIteration)
    {
        maxIteration = iteration;
    }
}

template <int N>
__global__
void upsample(
    const unsigned int numPixels,
    const float3& cameraPosition,
    const int2* texcoordsLow,
    int width,
    int height,
    int minimumIterations,
    float iterationAccelerator,
    bool mortonCurve,
    const float3* pixelDirs,
    float* pixelDepths,
    float* pixelDeltaDepths,
    int* pixelIterations)
{
    const unsigned int id = getPixelIndex();
    if (id >= numPixels)
    {
        return;
    }

    const int2& texcoord = texcoordsLow[id];
    const int x = texcoord.x;
    const int y = texcoord.y;

    float closestDepth = FLOAT_MAX;
    float closestDeltaDepth = 0;
    int closestIteration = 0;
    int maxIteration = 0;

    {
        const int halfWidth = width / 2;
        unsigned int neighbor;

        const int oddOffset = y % 2 == 0 ? 0 : 1;

        // Left pixel
        if(oddOffset || x > 0)
        {
            neighbor = getNeighborIndex(id, -oddOffset, 0, halfWidth, mortonCurve);
            getValuesFromClosestNeighbor(neighbor, closestDepth, closestDeltaDepth, closestIteration, maxIteration);
        }
        // Right pixel
        if(!oddOffset || x + 1 < halfWidth)
        {
            neighbor = getNeighborIndex(id, 1-oddOffset, 0, halfWidth, mortonCurve);
            getValuesFromClosestNeighbor(neighbor, closestDepth, closestDeltaDepth, closestIteration, maxIteration);
        }
        // Up pixel
        if(y > 0)
        {
            neighbor = getNeighborIndex(id, 0, -1, halfWidth, mortonCurve);
            getValuesFromClosestNeighbor(neighbor, closestDepth, closestDeltaDepth, closestIteration, maxIteration);
        }
        // Down pixel
        if (y + 1 < height)
        {
            neighbor = getNeighborIndex(id, 0, 1, halfWidth, mortonCurve);
            getValuesFromClosestNeighbor(neighbor, closestDepth, closestDeltaDepth, closestIteration, maxIteration);
        }
    }

    const unsigned int highResId = getIndex(y % 2 == 0 ? 2 * x + 1 : 2 * x, y, width, mortonCurve);
    const float3& dir = pixelDirs[highResId];
    float3 pos = cameraPosition + dir * closestDepth;

    int iterationLeft;
    float potential, dr;
    int res = evalMandelbulb<N>(pos, closestIteration, iterationLeft, potential, dr);
    if (res > 0)
    {
        while (res > 0)
        {
            closestDepth -= closestDeltaDepth;
            pos -= dir * closestDeltaDepth;
            res = evalMandelbulb<N>(pos, closestIteration, iterationLeft, potential, dr);
        }
    }
    else
    {
        closestDepth = closestDepth + closestDeltaDepth;
        pos += dir * closestDeltaDepth;
        int pixelIteration;
        rayMarchWithAcceleration<N>(
            pos,
            dir,
            iterationAccelerator,
            minimumIterations,
            -1,
            false,
            pixelIteration,
            closestIteration,
            closestDepth,
            closestDeltaDepth);
    }

    pixelDepths[highResId] = closestDepth;
    pixelDeltaDepths[highResId] = closestDeltaDepth;
    pixelIterations[highResId] = closestIteration;
}

// Upsample
template <int N>
__global__
void upsamplePhase1(
    const unsigned int numPixels,
    const float3& cameraPosition,
    int widthLow,
    int heightLow,
    int width,
    int minimumIterations,
    float iterationAccelerator,
    bool mortonCurve,
    const int2* texcoordsLow,
    const float* pixelDepthsLow,
    const float* pixelDeltaDepthsLow,
    const int* pixelIterationsLow,
    const float3* pixelDirs,
    float* pixelDepths,
    float* pixelDeltaDepths,
    int* pixelIterations)
{
    const unsigned int id = getPixelIndex();
    if (id >= numPixels)
    {
        return;
    }

    const int2& texcoord = texcoordsLow[id];
    const int baseX = texcoord.x;
    const int baseY = texcoord.y;

    float closestDepth = FLOAT_MAX;
    float closestDeltaDepth = 0;
    int closestIteration = 0;
    int maxIteration = 0;

    {
        unsigned int neighbor;

        for (int i = 0; i <= 1; ++i)
        {
            if (baseX + i <= widthLow)
            {
                continue;
            }

            for (int j = 0; j <= 1; ++j)
            {
                if (baseY + j <= heightLow)
                {
                    continue;
                }

                neighbor = getNeighborIndex(id, i, j, widthLow, mortonCurve);
                getValuesFromClosestNeighbor(neighbor, closestDepth, closestDeltaDepth, closestIteration, maxIteration);
            }
        }
    }

    const int x = 2 * baseX + 1;
    const int y = 2 * baseY + 1;
    const unsigned int highResId = getIndex(x, y, width, mortonCurve);
    const float3& dir = pixelDirs[highResId];
    float3 pos = cameraPosition + dir * closestDepth;

    int iterationLeft;
    float potential, dr;
    int res = evalMandelbulb<N>(pos, closestIteration, iterationLeft, potential, dr);
    if (res > 0)
    {
        while (res > 0)
        {
            closestDepth -= closestDeltaDepth;
            pos -= dir * closestDeltaDepth;
            res = evalMandelbulb<N>(pos, closestIteration, iterationLeft, potential, dr);
        }
    }
    else
    {
        closestDepth = closestDepth + closestDeltaDepth;
        pos += dir * closestDeltaDepth;
        int pixelIteration;
        rayMarchWithAcceleration<N>(
            pos,
            dir,
            iterationAccelerator,
            minimumIterations,
            -1,
            false,
            pixelIteration,
            closestIteration,
            closestDepth,
            closestDeltaDepth);
    }

    pixelDepths[highResId] = closestDepth;
    pixelDeltaDepths[highResId] = closestDeltaDepth;
    pixelIterations[highResId] = closestIteration;
}

template <int N>
__global__
void upsamplePhase2(
    const unsigned int numPixels,
    int width,
    int height,
    int minimumIterations,
    float iterationAccelerator,
    bool mortonCurve,
    const int2* texcoordsLow,
    const float3* pixelDirs,
    float* pixelDepths,
    float* pixelDeltaDepths,
    int* pixelIterations)
{
    const unsigned int id = getPixelIndex();
    if (id >= numPixels)
    {
        return;
    }

    const int2& texcoord = texcoordsLow[id];
    const int baseX = texcoord.x;
    const int baseY = texcoord.y;
    const int x = (baseY % 2 == 0) ?  (2 * baseX + 1) : (2 * baseX);
    const int y = 2 * baseY;
    const unsigned int highResId = getIndex(x, y, width, mortonCurve);

    float closestDepth = FLOAT_MAX;
    float closestDeltaDepth = 0;
    int closestIteration = 0;
    int maxIteration = 0;

    {
        unsigned int neighbor;

        // Left pixel
        if (x > 0)
        {
            neighbor = getNeighborIndex(highResId, -1, 0, width, mortonCurve);
            getValuesFromClosestNeighbor(neighbor, closestDepth, closestDeltaDepth, closestIteration, maxIteration);
        }
        // Right pixel
        if (x + 1 < width)
        {
            neighbor = getNeighborIndex(highResId, 1, 0, width, mortonCurve);
            getValuesFromClosestNeighbor(neighbor, closestDepth, closestDeltaDepth, closestIteration, maxIteration);
        }
        // Up pixel
        if (y > 0)
        {
            neighbor = getNeighborIndex(highResId, 0, -1, width, mortonCurve);
            getValuesFromClosestNeighbor(neighbor, closestDepth, closestDeltaDepth, closestIteration, maxIteration);
        }
        // Down pixel
        if (y + 1 < height)
        {
            neighbor = getNeighborIndex(highResId, 0, 1, width, mortonCurve);
            getValuesFromClosestNeighbor(neighbor, closestDepth, closestDeltaDepth, closestIteration, maxIteration);
        }
    }

    pixelDepths[highResId] = FLOAT_MAX;
    pixelDeltaDepths[highResId] = closestDeltaDepth;
    pixelIterations[highResId] = closestIteration;
}

template <int N>
__global__
void upsamplePhase3(
    const unsigned int numPixels,
    int width,
    int height,
    int minimumIterations,
    float iterationAccelerator,
    bool mortonCurve,
    const int2* texcoordsLow,
    const float3* pixelDirs,
    float* pixelDepths,
    float* pixelDeltaDepths,
    int* pixelIterations)
{
    const unsigned int id = getPixelIndex();
    if (id >= numPixels)
    {
        return;
    }

    const int2& texcoord = texcoordsLow[id];
    const int baseX = texcoord.x;
    const int baseY = texcoord.y;
    const int x = 2 * baseX;
    const int y = (baseX % 2 == 0) ? (2 * baseY + 1) : (2 * baseY);
    const unsigned int highResId = getIndex(x, y, width, mortonCurve);

    float closestDepth = FLOAT_MAX;
    float closestDeltaDepth = 0;
    int closestIteration = 0;
    int maxIteration = 0;

    {
        unsigned int neighbor;

        // Left pixel
        if (x > 0)
        {
            neighbor = getNeighborIndex(highResId, -1, 0, width, mortonCurve);
            getValuesFromClosestNeighbor(neighbor, closestDepth, closestDeltaDepth, closestIteration, maxIteration);
        }
        // Right pixel
        if (x + 1 < width)
        {
            neighbor = getNeighborIndex(highResId, 1, 0, width, mortonCurve);
            getValuesFromClosestNeighbor(neighbor, closestDepth, closestDeltaDepth, closestIteration, maxIteration);
        }
        // Up pixel
        if (y > 0)
        {
            neighbor = getNeighborIndex(highResId, 0, -1, width, mortonCurve);
            getValuesFromClosestNeighbor(neighbor, closestDepth, closestDeltaDepth, closestIteration, maxIteration);
        }
        // Down pixel
        if (y + 1 < height)
        {
            neighbor = getNeighborIndex(highResId, 0, 1, width, mortonCurve);
            getValuesFromClosestNeighbor(neighbor, closestDepth, closestDeltaDepth, closestIteration, maxIteration);
        }
    }

    pixelDepths[highResId] = FLOAT_MAX;
    pixelDeltaDepths[highResId] = closestDeltaDepth;
    pixelIterations[highResId] = closestIteration;
}

// Evaluate normal of pixels by simplified screen space calculation
__global__
void samplePseudoScreenSpaceNormals_kernel(
    const unsigned int numPixels,
    const int width,
    const int height,
    const int2* texcoords,
    const float3 cameraPos,
    bool mortonCurve,
    float3* pixelNormals)
{
    const unsigned int id = getPixelIndex();
    if (id >= numPixels)
    {
        return;
    }

    const unsigned int id3 = id * 3;

    // Get surface position
    float3 pos;
    pos.x = tex1Dfetch(s_positionTexture, id3);
    pos.y = tex1Dfetch(s_positionTexture, id3 + 1);
    pos.z = tex1Dfetch(s_positionTexture, id3 + 2);

    if (pos.x == FLOAT_MAX) return;

    // Get pixel coordinates
    const int2& texcoord = texcoords[id];
    const int x = texcoord.x;
    const int y = texcoord.y;

    float p1, p2;
    unsigned int neighborId;
    float3 nor = { 0.0f, 0.0f, 0.0f };

    // Right pixel
    if (x > 0)
    {
        neighborId = 3 * getNeighborIndex(id, -1, 0, width, mortonCurve);
        p1 = pos.x - tex1Dfetch(s_positionTexture, neighborId);
        p2 = pos.z - tex1Dfetch(s_positionTexture, neighborId + 2);
        nor.x += p2;
        nor.z += p1;
    }

    // Left pixel
    if (x + 1 < width)
    {
        neighborId = 3 * getNeighborIndex(id, 1, 0, width, mortonCurve);
        p1 = pos.x - tex1Dfetch(s_positionTexture, neighborId);
        p2 = pos.z - tex1Dfetch(s_positionTexture, neighborId + 2);
        nor.x -= p2;
        nor.z -= p1;
    }

    // Up pixel
    if (y > 0)
    {
        neighborId = 3 * getNeighborIndex(id, 0, -1, width, mortonCurve);
        p1 = pos.y - tex1Dfetch(s_positionTexture, neighborId + 1);
        p2 = pos.z - tex1Dfetch(s_positionTexture, neighborId + 2);
        nor.y += p2;
        nor.z += p1;
    }

    // Down pixel
    if (y + 1 < height)
    {
        neighborId = 3 * getNeighborIndex(id, 0, 1, width, mortonCurve);
        p1 = pos.y - tex1Dfetch(s_positionTexture, neighborId + 1);
        p2 = pos.z - tex1Dfetch(s_positionTexture, neighborId + 2);
        nor.y -= p2;
        nor.z -= p1;
    }

    pixelNormals[id] = normalize(nor);
}

// Evaluate normal of pixels by screen space depth
__global__
void sampleScreenSpaceNormals_kernel(
    const unsigned int numPixels,
    const int width,
    const int height,
    const int2* texcoords,
    const float3 cameraPos,
    bool mortonCurve,
    float3* pixelNormals)
{
    const unsigned int id = getPixelIndex();
    if (id >= numPixels)
    {
        return;
    }

    const unsigned int id3 = id * 3;

    // Get surface position
    float3 pos;
    pos.x = tex1Dfetch(s_positionTexture, id3);
    pos.y = tex1Dfetch(s_positionTexture, id3 + 1);
    pos.z = tex1Dfetch(s_positionTexture, id3 + 2);

    if (pos.x == FLOAT_MAX) return;

    // Get pixel coordinates
    const int2& texcoord = texcoords[id];
    const int x = texcoord.x;
    const int y = texcoord.y;

    unsigned int neighborId;
    float3 neighborPos1, neighborPos2;

    // Right pixel
    if (x != 0)
    {
        neighborId = 3 * getNeighborIndex(id, -1, 0, width, mortonCurve);
        neighborPos1.x = tex1Dfetch(s_positionTexture, neighborId);
        neighborPos1.y = tex1Dfetch(s_positionTexture, neighborId + 1);
        neighborPos1.z = tex1Dfetch(s_positionTexture, neighborId + 2);
    }
    else
    {
        neighborPos1 = pos;
    }

    // Left pixel
    if (x + 1 != width)
    {
        neighborId = 3 * getNeighborIndex(id, 1, 0, width, mortonCurve);
        neighborPos2.x = tex1Dfetch(s_positionTexture, neighborId);
        neighborPos2.y = tex1Dfetch(s_positionTexture, neighborId + 1);
        neighborPos2.z = tex1Dfetch(s_positionTexture, neighborId + 2);
    }
    else
    {
        neighborPos2 = pos;
    }

    const float3 dx = neighborPos2 - neighborPos1;

    // Up pixel
    if (y != 0)
    {
        neighborId = 3 * getNeighborIndex(id, 0, -1, width, mortonCurve);
        neighborPos1.x = tex1Dfetch(s_positionTexture, neighborId);
        neighborPos1.y = tex1Dfetch(s_positionTexture, neighborId + 1);
        neighborPos1.z = tex1Dfetch(s_positionTexture, neighborId + 2);
    }
    else
    {
        neighborPos1 = pos;
    }

    // Down pixel
    if (y + 1 != height)
    {
        neighborId = 3 * getNeighborIndex(id, 0, 1, width, mortonCurve);
        neighborPos2.x = tex1Dfetch(s_positionTexture, neighborId);
        neighborPos2.y = tex1Dfetch(s_positionTexture, neighborId + 1);
        neighborPos2.z = tex1Dfetch(s_positionTexture, neighborId + 2);
    }
    else
    {
        neighborPos2 = pos;
    }

    const float3 dy = neighborPos2 - neighborPos1;

    pixelNormals[id] = normalize(cross(dx, dy));
}

// Apply SSAO effect
__global__
void applySSAO_kernel(
    const unsigned int numPixels,
    const int width,
    const int height,
    const int2* texcoords,
    bool mortonCurve,
    float3* pixelColors)
{
    const unsigned int id = getPixelIndex();
    if (id >= numPixels)
    {
        return;
    }

    const int2& cord = texcoords[id];
    const int x = cord.x;
    const int y = cord.y;

    float depth = tex1Dfetch(s_depthTexture, id);

    unsigned int shieldCount = 1;
    unsigned int neighborId;
    float neighborDepth;

    for (int i = -3; i <= 3; ++i)
    {
        if ((x == 0 && i < 0) || (x + 1 == width && i > 0))
        {
            continue;
        }
        for (int j = -3; j <= 3; ++j)
        {
            if (i == 0 && j == 0)
            {
                continue;
            }

            if (i * i + j * j > 9)
            {
                continue;
            }

            if ((y == 0 && j < 0) || (y + 1 == height && j > 0))
            {
                continue;
            }

            neighborId = getNeighborIndex(id, i, j, width, mortonCurve);
            neighborDepth = tex1Dfetch(s_depthTexture, neighborId);
            if (depth > neighborDepth)
            {
                ++shieldCount;
            }
        }
    }

    float shield = 1.f / (float)shieldCount * 9.f + 0.3f;

    // Clamp shield
    if (shield > 1.0f)
    {
        return;
    }

    // Modify the color
    pixelColors[id].x *= shield;
    pixelColors[id].y *= shield;
    pixelColors[id].z *= shield;
}

template <int N>
__global__
void castShadow(
    const unsigned int numPixels,
    const unsigned int minimumIterations,
    const float3* pixelPositions,
    const float* pixelDepths,
    const float3* lightPositions,
    const unsigned int numLights,
    float3* pixelColors)
{
    const unsigned int id = getPixelIndex();
    if (id >= numPixels)
    {
        return;
    }

    if (id < numPixels)
    {
        const float3& pixelPos = pixelPositions[id];
        const float depth = pixelDepths[id];
        float deltaDepth = 0.0005f * depth;

        bool lit = false;

        for (int l = 0; l < numLights; ++l)
        {
            const float3 dir = lightPositions[l] - pixelPos;
            float3 pos = pixelPos + dir * deltaDepth;

            float count = 0;
            int i;
            for (i = 0; i < 25; i++)
            {
                if (dot(pos, pos) > s_sphereRadiusSquared)
                {
                    lit = true;
                    break;
                }

                const int currentIteration = getNumIterations(depth, 0, minimumIterations);
                int iterationLeft;
                float residual, potential;
                if (evalMandelbulb<N>(pos, currentIteration, iterationLeft, potential, residual))
                {
                    break;
                }
                pos = pos + dir * deltaDepth;

                count += 0.0002f;
                deltaDepth *= pow(1000.0f, count);
            }

            if (i == 25)
            {
                lit = true;
                break;
            }
        }

        if (!lit)
        {
            pixelColors[id] *= 0.5f;
        }
    }
}

// Calculate color of pixels based on position, normal and lights
__global__
void setColorFromPos(
    const unsigned int numPixels,
    const float3* pixelPositions,
    const float* pixelDepths,
    const float3* pixelNormals,
    const float3 cameraPosition,
    const float3* lightPositions,
    const unsigned int numLights,
    bool useNormal,
    bool colorfulMode,
    float3* pixelColors)
{
    const unsigned int id = getPixelIndex();
    if (id >= numPixels)
    {
        return;
    }

    float3 col;
    const float3& pos = pixelPositions[id];
    if (pixelDepths[id] != FLOAT_MAX)
    {
        const float gain = 0.2f;
        const float ampl = 1.2f;
        col = (pos + ampl) * gain;

        if (useNormal)
        {
            const float3& normal = pixelNormals[id];
            const float div = dot(normal, pos);
            const float divSquared = div * div;

            float diffuse = 0;
            float specular = 0;

            for (int i = 0; i < numLights; ++i)
            {
                const float3& lightPosition = lightPositions[i];

                diffuse += 0.5f * max(dot(normal, normalize(lightPosition - pos)), 0.0f);
                specular += pow(max(dot(normal, normalize(cameraPosition - lightPosition)), 0.0f), 8.0f);
            }

            diffuse *= 2.f;
            specular *= 0.5f;

            const float ambient = 1.0f;
            const float ambientAndDiffuse = ambient + diffuse;

            if (colorfulMode)
            {
                float col1 = 1.0f, col2 = 1.0f;
                if (div > 0.0f)
                {
                    col2 += divSquared;
                }
                else
                {
                    col1 += divSquared;
                }
                const float col3 = 0.003f / (divSquared * divSquared + 0.01f);

                col.x = col.x * ambientAndDiffuse * col1 + specular;
                col.y = col.y * ambientAndDiffuse + specular + col3;
                col.z = col.z * ambientAndDiffuse * col2 + specular;
            }
            else
            {
                col.x = col.x * ambientAndDiffuse + specular;
                col.y = col.y * ambientAndDiffuse + specular;
                col.z = col.z * ambientAndDiffuse + specular;
            }
        }

        // Visualize normal as color
        //if (useNormal)
        //{
        //    const float3& normal = pixelNormals[id];
        //    col = normal;
        //}
    }
    else
    {
        col.x = 0.0f;
        col.y = 0.0f;
        col.z = 0.0f;
    }

    pixelColors[id] = col;

}

// Convert floating point [0:1] color into 256 color
__global__
void colFloatToByte(
    const unsigned int numPixels,
    const float3* pixelColorsFloat,
    int width,
    bool mortonCurve,
    unsigned char* pixelColorsUChar)
{
    const unsigned int id = getPixelIndex();
    if (id < numPixels)
    {
        const float3 col = pixelColorsFloat[id];
        const float r = col.x > 1.f ? 1.0 : (col.x < 0.f ? 0.f : col.x);
        const float g = col.y > 1.f ? 1.0 : (col.y < 0.f ? 0.f : col.y);
        const float b = col.z > 1.f ? 1.0 : (col.z < 0.f ? 0.f : col.z);

        unsigned int outId;
        if (mortonCurve)
        {
            unsigned short x, y;
            decodeMortonCurve(id, x, y);
            outId = x + width * y;
        }
        else
        {
            outId = id;
        }

        unsigned char* colUChar = &pixelColorsUChar[outId * 3];
        colUChar[0] = (unsigned char)(255 * r);
        colUChar[1] = (unsigned char)(255 * g);
        colUChar[2] = (unsigned char)(255 * b);
    }
}

#ifdef ENABLE_DOUBLE_PRECISION
#include "mandelbulbRendererDouble.cu"
#endif

} // namespace MandelbulbCudaKernel

// Costructor
MandelbulbRenderer::MandelbulbRenderer(unsigned int width, unsigned int height)
    : m_width(width)
    , m_height(height)
    , m_numPixels(width * height)
{
    cudaSetDevice(0);
    cudaThreadSynchronize();

    computeGridSize(m_numPixels, 256, m_numBlocks, m_numThreads);
    computeGridSize(m_numPixels / 2, 256, m_numBlocksForLow, m_numThreadsForLow);

    allocateMemory();

    MandelbulbCudaKernel::setTexcoords<<<m_numBlocks, m_numThreads>>>(
        m_numPixels,
        (int)m_width,
        (int)m_height,
        m_mortonCurve,
        m_texcoords);
}

// Destructor
MandelbulbRenderer::~MandelbulbRenderer()
{
    freeMemory();
    freeAdaptiveIterationsMemory();
    freeUpsamplingMemory();
}

// Update camera information
void MandelbulbRenderer::updateCamera(const float3& pos, const float3& forward, const float3& up, const float3& side)
{
    m_cameraPosition = pos;
    m_cameraForward = forward;
    m_cameraUp = up;
    m_cameraSide = side;
    setDirty();
}

// Set pixel angle
void MandelbulbRenderer::setPixelAngle(float angle)
{
    if (angle > 0)
    {
        m_pixelAngle = angle;
    }
}

// Calculate mandelbulb and set colors to pixel buffer
void MandelbulbRenderer::createMandelbulb(unsigned char* pixcelColorsHost)
{
    using namespace MandelbulbCudaKernel;

    clock_t timer;
    if (m_profile)
    {
        timer = clock();
    }

    // Set initial iteration count
    {
        startTimer("Initialization");

        // Make sure that iteration counts is bigger than minimum counts
        if (m_initialIterations < m_minimumIterations)
        {
            m_initialIterations = m_minimumIterations;
        }

        // Run mandelbulb at the camera position on device once and determine the initial iteration counts
        int iterationLeft;
        evalMandelbulbOnHost<8>(m_cameraPosition, s_numIterationsOnHost, iterationLeft);

        if (iterationLeft > 0)
        {
            int numIteration = s_numIterationsOnHost - iterationLeft + 1;
            if (numIteration > m_initialIterations)
            {
                m_initialIterations = numIteration;
            }
        }

        endTimer();
    }

    {
        startTimer("Set Pixel Direction");

        // Set direction of rays
        setPixelDirection <<<m_numBlocks, m_numThreads>>> (
            m_numPixels,
            (int)m_width / 2,
            (int)m_height / 2,
            m_texcoords,
            m_cameraForward,
            m_cameraUp,
            m_cameraSide,
            m_pixelAngle,
            m_pixelDirs);

        if (m_adaptiveIterationCount && m_upsampling)
        {
            setPixelDirectionLow <<<m_numBlocksForLow, m_numBlocksForLow>>> (
                m_numPixels / 2,
                (int)m_width / 2,
                (int)m_height / 2,
                m_texcoordsLow,
                m_cameraForward,
                m_cameraUp,
                m_cameraSide,
                m_pixelAngle,
                m_pixelDirsLow);
        }

        cudaThreadSynchronize();
        endTimer();
    }

    {
        startTimer("Ray March");

#ifdef ENABLE_DOUBLE_PRECISION
        // Perform mandelbulb
        if (m_doublePrecision && !m_upsampling)
        {
            raymarchMandelbulbD_kernel<8> <<<m_numBlocks, m_numThreads>>> (
                m_numPixels,
                m_pixelDirs,
                m_cameraPosition,
                m_initialDeltaStep,
                m_iterationAccelerator,
                m_initialIterations,
                false,
                m_pixelIterations,
                m_pixelDepths,
                m_pixelDeltaDepths);
        }
        else
#endif
        {
            int numPixels = m_upsampling ? m_numPixels / 2 : m_numPixels;
            int numBlocks = m_upsampling ? m_numBlocksForLow : m_numBlocks;
            int numThreads = m_upsampling ? m_numThreadsForLow : m_numThreads;
            float3* dirs = m_upsampling ? m_pixelDirsLow : m_pixelDirs;
            int* iterations = m_upsampling ? m_pixelIterationsLow : m_pixelIterations;
            float* depths = m_upsampling ? m_pixelDepthsLow : m_pixelDepths;
            float* deltaDepths = m_upsampling ? m_pixelDeltaDepthsLow : m_pixelDeltaDepths;

            raymarchMandelbulb_kernel<8> <<<numBlocks, numThreads>>> (
                numPixels,
                dirs,
                m_cameraPosition,
                m_initialDeltaStep,
                m_iterationAccelerator,
                m_initialIterations,
                m_distanceEstimation,
                iterations,
                m_numRayMarchSteps,
                depths,
                deltaDepths);
        }

        cudaThreadSynchronize();
        endTimer();
    }

    if (m_adaptiveIterationCount && !m_doublePrecision)
    {
        startTimer("Adaptive Iteration");

        // Drill down the surface by adaptive iteration counts

        int numPixels      = m_upsampling ? m_numPixels / 2 : m_numPixels;
        int numBlocks      = m_upsampling ? m_numBlocksForLow : m_numBlocks;
        int numThreads     = m_upsampling ? m_numThreadsForLow : m_numThreads;
        int width          = m_upsampling ? (int)m_width / 2 : (int)m_width;
        int height         = m_upsampling ? (int)m_height / 2 : (int)m_height;
        int2* texcoords    = m_upsampling ? m_texcoordsLow : m_texcoords;
        float3* dirs       = m_upsampling ? m_pixelDirsLow : m_pixelDirs;
        float* depths      = m_upsampling ? m_pixelDepthsLow : m_pixelDepths;
        float* deltaDepths = m_upsampling ? m_pixelDeltaDepthsLow : m_pixelDeltaDepths;
        float* depthDiffs  = m_upsampling ? m_pixelDepthDiffsLow : m_pixelDepthDiffs;
        float* depthTmp    = m_upsampling ? m_pixelDepthsTmpLow : m_pixelDepthsTmp;
        int* iterations    = m_upsampling ? m_pixelIterationsLow : m_pixelIterations;

        cudaBindTexture(0, s_depthDiffTexture, depthDiffs, numPixels * sizeof(float));
        cudaBindTexture(0, s_depthTexture, depths, numPixels * sizeof(float));
        cudaBindTexture(0, s_iterationTexture, iterations, numPixels * sizeof(int));

        for (int i = 0; i < m_numDrillingIterations; ++i)
        {
            adaptiveIteration_kernel<8><<<numBlocks, numThreads >>>(
                numPixels,
                dirs,
                m_cameraPosition,
                m_minimumIterations,
                m_iterationToDeltaStepRate,
                m_distanceEstimation,
                iterations,
                depths,
                depthTmp,
                deltaDepths,
                depthDiffs
                );

            compareDepthDiffs_kernel<<<numBlocks, numThreads >>>(
                numPixels,
                width,
                height,
                m_laplacianThreshold,
                depthDiffs,
                depthTmp,
                texcoords,
                m_iterationToDeltaStepRate,
                m_mortonCurve,
                m_upsampling,
                iterations,
                depths,
                deltaDepths
                );

            cudaThreadSynchronize();
        }

        endTimer();

        {
            startTimer("Finalize Adaptive Iteration");

            finalizeIteration_kernel <<<numBlocks, numThreads>>> (
                numPixels,
                iterations
                );

            cudaThreadSynchronize();
            endTimer();
        }

        if (m_upsampling)
        {
            startTimer("Upsample");

            copyFromLowToHigh<<<m_numBlocksForLow, m_numThreadsForLow>>>(
                numPixels,
                (int)m_width,
                m_mortonCurve,
                m_texcoordsLow,
                m_pixelDepthsLow,
                m_pixelDeltaDepthsLow,
                m_pixelIterationsLow,
                m_pixelDepths,
                m_pixelDeltaDepths,
                m_pixelIterations
                );

            cudaBindTexture(0, s_deltaDepthTexture, deltaDepths, numPixels * sizeof(float));

            cudaThreadSynchronize();

            upsample<8><<<m_numBlocksForLow, m_numThreadsForLow>>>(
                numPixels,
                m_cameraPosition,
                m_texcoordsLow,
                (int)m_width,
                (int)m_height,
                m_minimumIterations,
                m_iterationAccelerator,
                m_mortonCurve,
                m_pixelDirs,
                m_pixelDepths,
                m_pixelDeltaDepths,
                m_pixelIterations
                );

            upsamplePhase1<8><<<m_numBlocksForLow, m_numThreadsForLow>>>(
                numPixels,
                m_cameraPosition,
                (int)m_width / 2,
                (int)m_height / 2,
                (int)m_width,
                m_minimumIterations,
                m_iterationAccelerator,
                m_mortonCurve,
                m_texcoordsLow,
                m_pixelDepthsLow,
                m_pixelDeltaDepthsLow,
                m_pixelIterationsLow,
                m_pixelDirs,
                m_pixelDepths,
                m_pixelDeltaDepths,
                m_pixelIterations
                );

            cudaThreadSynchronize();

            cudaUnbindTexture(s_depthTexture);
            cudaUnbindTexture(s_deltaDepthTexture);
            cudaUnbindTexture(s_iterationTexture);

            cudaBindTexture(0, s_depthTexture, m_pixelDepths, m_numPixels * sizeof(float));
            cudaBindTexture(0, s_deltaDepthTexture, m_pixelDeltaDepths, m_numPixels * sizeof(float));
            cudaBindTexture(0, s_iterationTexture, m_pixelIterations, m_numPixels * sizeof(int));

            upsamplePhase2<8><<<m_numBlocksForLow, m_numThreadsForLow>>>(
                numPixels,
                (int)m_width,
                (int)m_height,
                m_minimumIterations,
                m_iterationAccelerator,
                m_mortonCurve,
                m_texcoordsLow,
                m_pixelDirs,
                m_pixelDepths,
                m_pixelDeltaDepths,
                m_pixelIterations);

            upsamplePhase3<8><<<m_numBlocksForLow, m_numThreadsForLow>>>(
                numPixels,
                (int)m_width,
                (int)m_height,
                m_minimumIterations,
                m_iterationAccelerator,
                m_mortonCurve,
                m_texcoordsLow,
                m_pixelDirs,
                m_pixelDepths,
                m_pixelDeltaDepths,
                m_pixelIterations);

            cudaUnbindTexture(s_depthTexture);
            cudaUnbindTexture(s_deltaDepthTexture);
            cudaUnbindTexture(s_iterationTexture);

            endTimer();
        }
        else
        {
            cudaUnbindTexture(s_iterationTexture);
            cudaUnbindTexture(s_depthTexture);
        }

        cudaUnbindTexture(s_depthDiffTexture);
    }

    {
        startTimer("Bisection Search");

        // Perform bisection search to determine the final position of the surface
#ifdef ENABLE_DOUBLE_PRECISION
        if (m_doublePrecision && !m_upsampling)
        {
            binaryPartitionSearchD<8> << <m_numBlocks, m_numThreads >> > (
                m_numPixels,
                m_pixelDirs,
                m_pixelDeltaDepths,
                m_cameraPosition,
                m_pixelIterations,
                m_pixelPositions,
                m_pixelDepths);
        }
        else
#endif
        {
            binaryPartitionSearch<8> << <m_numBlocks, m_numThreads >> > (
                m_numPixels,
                m_pixelDirs,
                m_pixelDeltaDepths,
                m_cameraPosition,
                m_pixelIterations,
                m_pixelPositions,
                m_pixelDepths);
        }

        cudaThreadSynchronize();
        endTimer();
    }

    {
        startTimer("Update Parameters");

        // Update initial iteration counts of the next step by taking minimum iterations of this step
        m_initialIterations = 0;
        cpyDeviceToHost((void*)m_pixelIterationsHost, (void*)(m_pixelIterations), m_numPixels * sizeof(int));
        int m_initialIterations = INT_MAX;
        for (int i = 0; i < (int)m_numPixels; ++i)
        {
            if (m_pixelIterationsHost[i] > 0 && m_initialIterations > m_pixelIterationsHost[i])
            {
                m_initialIterations = m_pixelIterationsHost[i];
            }
        }

        // Update focal depth and initial delta step of the next step based on the result of this step
        if (m_needRecalculate)
        {
            float centerDepth;
            int offset;
            if (m_mortonCurve)
            {
                offset = (int)encodeMortonCurveHost(m_width / 2, m_height / 2);
            }
            else
            {
                offset = m_numPixels / 2 - m_width / 2;
            }
            cpyDeviceToHost((void*)&centerDepth, (void*)(m_pixelDepths + offset), sizeof(float));
            if (centerDepth > s_minDepth && centerDepth < s_maxDepth)
            {
                m_focalDepth = centerDepth;
                m_initialDeltaStep = centerDepth * m_depthToDeltaStepRate;
            }

            m_needRecalculate = false;
        }

        endTimer();
    }

    // Calculate normal
    if(m_normalMode != NoNormal)
    {
        startTimer("Calculate Normal");

        cudaBindTexture(0, s_positionTexture, m_pixelPositions, m_numPixels * sizeof(float3));

        switch (m_normalMode)
        {
        case PseudoScreenSpace:
            samplePseudoScreenSpaceNormals_kernel<<<m_numBlocks, m_numThreads>>>(
                m_numPixels,
                (int)m_width,
                (int)m_height,
                m_texcoords,
                m_cameraPosition,
                m_mortonCurve,
                m_pixelNormals);
            break;
        case ScreenSpace:
            sampleScreenSpaceNormals_kernel<<<m_numBlocks, m_numThreads>>>(
                m_numPixels,
                (int)m_width,
                (int)m_height,
                m_texcoords,
                m_cameraPosition,
                m_mortonCurve,
                m_pixelNormals);
            break;
        }

        cudaUnbindTexture(s_positionTexture);

        cudaThreadSynchronize();
        endTimer();
    }

    const unsigned int numLights = (unsigned int)m_lightPositionsHost.size();

    {
        startTimer("Set Color");

        if (!m_lightPositions && numLights > 0)
        {
            allocateArray((void**)&m_lightPositions, numLights * sizeof(float3));
            cpyHostToDevice((void*)m_lightPositions, (void*)(&m_lightPositionsHost.front()), numLights * sizeof(float3));
        }

        // Set colors
        setColorFromPos << <m_numBlocks, m_numThreads >> > (
            m_numPixels,
            m_pixelPositions,
            m_pixelDepths,
            m_pixelNormals,
            m_cameraPosition,
            m_lightPositions,
            numLights,
            m_normalMode != NoNormal,
            m_coloringMode == Colorful,
            m_pixelColorsFloat);

        cudaThreadSynchronize();
        endTimer();
    }

    // Cast shadow
    if(m_castShadow)
    {
        startTimer("Cast Shadow");

        castShadow<8> << <m_numBlocks, m_numThreads >> >(
            m_numPixels,
            m_minimumIterations,
            m_pixelPositions,
            m_pixelDepths,
            m_lightPositions,
            numLights,
            m_pixelColorsFloat);

        cudaThreadSynchronize();
        endTimer();
    }

    // Apply SSAO
    if(m_ssaoEnabled)
    {
        startTimer("SSAO");

        cudaBindTexture(0, s_depthTexture, m_pixelDepths, m_numPixels * sizeof(float));

        applySSAO_kernel<<<m_numBlocks, m_numThreads >>>(
            m_numPixels,
            (int)m_width,
            (int)m_height,
            m_texcoords,
            m_mortonCurve,
            m_pixelColorsFloat
            );

        cudaUnbindTexture(s_depthTexture);

        cudaThreadSynchronize();
        endTimer();
    }

    // Convert color from float to byte
    {
        startTimer("Prepare Result");

        colFloatToByte<<<m_numBlocks, m_numThreads>>>(m_numPixels, m_pixelColorsFloat, (int)m_width, m_mortonCurve, pixcelColorsHost);

        cudaThreadSynchronize();
        endTimer();
    }

    if (m_profile)
    {
        printf("--------------------\n");
        clock_t endTimer = clock();
        printMilliSeconds(timer, endTimer, "Total");
        printf("--------------------\n");

        m_profile = false;
    }
}


void MandelbulbRenderer::setMinimumIterations(int minItr)
{
    m_minimumIterations = minItr;
    m_initialIterations = 0;
}

void MandelbulbRenderer::setIterationAccelerator(float factor)
{
    m_iterationAccelerator = factor;
    m_initialIterations = 0;
}

void MandelbulbRenderer::enableAdaptiveIterations(bool enable)
{
    m_adaptiveIterationCount = enable;
    if (m_adaptiveIterationCount)
    {
        m_minimumIterations = 3;
        allocateAdaptiveIterationsMemory();
    }
    else
    {
        freeAdaptiveIterationsMemory();
    }

    m_initialIterations = 0;
}

void MandelbulbRenderer::enableDistanceEstimation(bool enable)
{
    m_distanceEstimation = enable;
}

void MandelbulbRenderer::enableUpsampling(bool enable)
{
    m_upsampling = enable;
    if (m_upsampling)
    {
        allocateUpsamplingMemory();
    }
    else
    {
        freeUpsamplingMemory();
    }
}

void MandelbulbRenderer::enableMortonCurveIndexing(bool enable)
{
    m_mortonCurve = enable;

    // Set up pixel coordinates
    MandelbulbCudaKernel::setTexcoords<<<m_numBlocks, m_numThreads>>>(
        m_numPixels,
        (int)m_width,
        (int)m_height,
        m_mortonCurve,
        m_texcoords);
}

void MandelbulbRenderer::addLight(const float3& pos)
{
    m_lightPositionsHost.push_back(pos);
    if (m_lightPositions)
    {
        freeArray((void**)m_lightPositions);
    }
}


void MandelbulbRenderer::setDepthToDeltaStepRate(float rate)
{
    m_depthToDeltaStepRate = rate;
    m_initialDeltaStep = m_focalDepth * m_depthToDeltaStepRate;
}

// Allocate buffers
void MandelbulbRenderer::allocateMemory()
{
    const int pixelsTimeFloat3 = m_numPixels * sizeof(float3);
    const int pixelsTimeFloat = m_numPixels * sizeof(float);

    allocateArray((void**)&m_pixelColorsFloat, pixelsTimeFloat3);
    allocateArray((void**)&m_pixelDirs, pixelsTimeFloat3);
    allocateArray((void**)&m_pixelNormals, pixelsTimeFloat3);
    allocateArray((void**)&m_pixelPositions, pixelsTimeFloat3);
    allocateArray((void**)&m_texcoords, m_numPixels * sizeof(int2));
    allocateArray((void**)&m_pixelDepths, pixelsTimeFloat);
    allocateArray((void**)&m_pixelDeltaDepths, pixelsTimeFloat);
    allocateArray((void**)&m_pixelIterations, m_numPixels * sizeof(int));
    allocateArray((void**)&m_numRayMarchSteps, m_numPixels * sizeof(int));

    m_pixelIterationsHost = new int[m_numPixels];

    if (m_adaptiveIterationCount)
    {
        allocateAdaptiveIterationsMemory();

        if (m_upsampling)
        {
            allocateUpsamplingMemory();
        }
    }
}

void MandelbulbRenderer::allocateAdaptiveIterationsMemory()
{
    if (m_pixelDepthDiffs)
    {
        // Already allocated
        return;
    }

    const int pixelsTimeFloat = m_numPixels * sizeof(float);
    allocateArray((void**)&m_pixelDepthDiffs, pixelsTimeFloat);
    allocateArray((void**)&m_pixelDepthsTmp, pixelsTimeFloat);
}

void MandelbulbRenderer::allocateUpsamplingMemory()
{
    if (m_pixelDirsLow)
    {
        // Already allocated
        return;
    }

    freeAdaptiveIterationsMemory();

    int numPixels = m_numPixels / 2;
    const int pixelsTimeFloat3 = numPixels * sizeof(float3);
    const int pixelsTimeFloat = numPixels * sizeof(float);

    allocateArray((void**)&m_pixelDirsLow, pixelsTimeFloat3);
    allocateArray((void**)&m_pixelDepthsLow, pixelsTimeFloat);
    allocateArray((void**)&m_pixelDeltaDepthsLow, pixelsTimeFloat);
    allocateArray((void**)&m_pixelIterationsLow, numPixels * sizeof(int));
    allocateArray((void**)&m_pixelDepthDiffsLow, pixelsTimeFloat);
    allocateArray((void**)&m_pixelDepthsTmpLow, pixelsTimeFloat);
    allocateArray((void**)&m_texcoordsLow, numPixels * sizeof(int2));

    MandelbulbCudaKernel::setTexcoords<<<m_numBlocksForLow, m_numThreadsForLow>>>(
        numPixels,
        (int)m_width / 2,
        (int)m_height,
        m_mortonCurve,
        m_texcoordsLow);
}

void MandelbulbRenderer::freeMemory()
{
    freeArray((void**)&m_pixelColorsFloat);
    freeArray((void**)&m_pixelDirs);
    freeArray((void**)&m_pixelNormals);
    freeArray((void**)&m_pixelPositions);
    freeArray((void**)&m_texcoords);
    freeArray((void**)&m_pixelDepths);
    freeArray((void**)&m_pixelDeltaDepths);
    freeArray((void**)&m_pixelIterations);
    freeArray((void**)&m_numRayMarchSteps);

    delete[] m_pixelIterationsHost;
    m_pixelIterationsHost = nullptr;
}

void MandelbulbRenderer::freeAdaptiveIterationsMemory()
{
    if (m_pixelDepthDiffs)
    {
        freeArray((void**)&m_pixelDepthDiffs);
        freeArray((void**)&m_pixelDepthsTmp);
    }
}

void MandelbulbRenderer::freeUpsamplingMemory()
{
    if (m_pixelDirsLow)
    {
        freeArray((void**)&m_pixelDirsLow);
        freeArray((void**)&m_pixelDepthsLow);
        freeArray((void**)&m_pixelDeltaDepthsLow);
        freeArray((void**)&m_pixelDepthDiffsLow);
        freeArray((void**)&m_pixelDepthsTmpLow);
        freeArray((void**)&m_texcoordsLow);
    }
}

void MandelbulbRenderer::startTimer(const char* timerName)
{
    if (m_profile)
    {
        m_startTimer = clock();
        m_timerName = timerName;
    }
}

void MandelbulbRenderer::endTimer() const
{
    if (m_profile)
    {
        clock_t endTimer = clock();
        printMilliSeconds(m_startTimer, endTimer, m_timerName);
    }
}

void MandelbulbRenderer::printMilliSeconds(const clock_t& c0, const clock_t& c1, const char* name) const
{
    const clock_t deltaClock = c1 - c0;
    float msec = deltaClock * 1000.f / (float)CLOCKS_PER_SEC;
    printf("%s : %f msec \n", name, msec);
}
