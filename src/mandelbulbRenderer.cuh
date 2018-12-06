//
// Copyright (c) 2018 Kohei Nagasawa
// Read LICENSE.md for license condition of this software
//

#include <helper_math.h>

// MandelbulbRenderer evaluates Mandelbulb set in 3D space and render it
class MandelbulbRenderer
{
public:

    // Constructor
    MandelbulbRenderer(unsigned int width, unsigned int height);

    // Destructor
    ~MandelbulbRenderer();

    // Normal mode to shade Mandelbulb set
    enum NormalMode
    {
        NoNormal = 0,
        PseudoScreenSpace,
        ScreenSpace,
        DistanceField,
    };

    // Color mode to decide final coloring of Mandelbulb shape
    enum ColoringMode
    {
        Normal,
        Colorful,
    };

public:

    // Calculate Mandelbulb and set colors to pixel buffer
    void createMandelbulb(unsigned char* pixelColorsHost);

    // Perform ray marching once on CPU from the center of view
    void rayMarchOnHost(float deltaDepth = -1);

    // Toggle profiling
    void profile() { m_profile = true; }

public:

    // Update camera information
    void updateCamera(const float3& pos, const float3& forward, const float3& up, const float3& side);

    // Set pixel angle
    void setPixelAngle(float angle);

public:

    // Getters / setters for parameters used for Mandelbulb evaluation

    void setMinimumIterations(int minItr);

    inline bool isAdaptiveIterationsEnabled() const { return m_adaptiveIterationCount; }
    void enableAdaptiveIterations(bool enable);

    void enableDistanceEstimation(bool enable);

    void enableUpsampling(bool enable);

    inline void setLaplacianThreshold(float lap) { m_laplacianThreshold = lap; }

    inline void setNumDrillingIterations(int itr) { m_numDrillingIterations = itr; }

    inline void setIterationToDeltaStepRate(float rate) { m_iterationToDeltaStepRate = rate; }

    void setDepthToDeltaStepRate(float rate);

    void setIterationAccelerator(float factor);

    inline void enableDoublePrecision(bool enable) { m_doublePrecision = enable; }

    inline float getInitialDeltaStep() const { return m_initialDeltaStep; }
    inline void setInitialDeltaStep(float ds) { m_initialDeltaStep = ds; }

    inline float getFocalDepth() const { return m_focalDepth; }

    void enableMortonCurveIndexing(bool enable);

public:

    // Getters / setters for parameters used for rendering and coloring

    inline void setNormalMode(NormalMode mode) { m_normalMode = mode; }
    inline NormalMode isNormalEnabled() const { return m_normalMode; }

    inline void setColoringMode(ColoringMode mode) { m_coloringMode = mode; }

    inline void enableSSAO(bool enable) { m_ssaoEnabled = enable; }
    inline bool isSSAOEnabled() const { return m_ssaoEnabled; }

    inline void setLight1Pos(const float3& pos) { m_light1Pos = pos; }
    inline void setLight2Pos(const float3& pos) { m_light2Pos = pos; }

private:

    // Mark as dirty and have Mandelbulb be evaluated again in the next step
    inline void setDirty() { m_needRecalculate = true; }

    // Allocate buffers
    void allocateMemory();
    void allocateAdaptiveIterationsMemory();
    void allocateUpsamplingMemory();

    // Free buffers
    void freeMemory();
    void freeAdaptiveIterationsMemory();
    void freeUpsamplingMemory();

    // Functions for profiling
    void startTimer(const char* timerName);
    void endTimer() const;
    void printMilliSeconds(const clock_t& c0, const clock_t& c1, const char* name) const;

private:

    // Data buffers on device
    float3* m_pixelColorsFloat = nullptr;
    float3* m_pixelDirs = nullptr;
    float3* m_pixelNormals = nullptr;
    float3* m_pixelPositions = nullptr;
    float* m_pixelDepths = nullptr;
    float* m_pixelDeltaDepths = nullptr;
    int2* m_texcoords = nullptr;
    int* m_pixelIterations = nullptr;
    float* m_pixelDepthDiffs = nullptr;
    float* m_pixelDepthsTmp = nullptr;
    int* m_numRayMarchSteps = nullptr;

    // Data buffers of downscaled version
    bool m_upsampling = false;
    float3* m_pixelDirsLow = nullptr;
    float* m_pixelDepthsLow = nullptr;
    float* m_pixelDeltaDepthsLow = nullptr;
    float* m_pixelDepthDiffsLow = nullptr;
    float* m_pixelDepthsTmpLow = nullptr;
    int* m_pixelIterationsLow = nullptr;
    int2* m_texcoordsLow = nullptr;

    // Data buffer on host
    int* m_pixelIterationsHost = nullptr;

    // Camera status
    float3 m_cameraPosition;
    float3 m_cameraForward;
    float3 m_cameraUp;
    float3 m_cameraSide;

    // Window info
    unsigned int m_numPixels = 0;
    const unsigned int m_width;
    const unsigned int m_height;
    float m_pixelAngle = 0;

    // Mandelbulb evaluation parameters
    int m_initialIterations = 0;
    int m_minimumIterations;
    int m_numDrillingIterations;
    bool m_doublePrecision = false;
    bool m_adaptiveIterationCount = true;
    bool m_mortonCurve = false;
    bool m_distanceEstimation = false;
    float m_laplacianThreshold;
    float m_iterationToDeltaStepRate;
    float m_depthToDeltaStepRate;
    float m_iterationAccelerator;
    float m_initialDeltaStep;
    float m_focalDepth = 1.5f;

    // Rendering parameters
    NormalMode m_normalMode = ScreenSpace;
    ColoringMode m_coloringMode = Normal;
    float3 m_light1Pos = { 0, 0, 0 };
    float3 m_light2Pos = { 0, 0, 0 };
    bool m_ssaoEnabled = true;

    // True if Mandelbulb needs to be evaluated again in the next step
    bool m_needRecalculate = true;

    // Members for profiling
    bool m_profile = false;
    clock_t m_startTimer;
    const char* m_timerName;

    // The number of blocks and threads of GPU core
    unsigned int m_numBlocks = 0;
    unsigned int m_numThreads = 0;
    unsigned int m_numBlocksForLow = 0;
    unsigned int m_numThreadsForLow = 0;
};
