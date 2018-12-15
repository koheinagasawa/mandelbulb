//
// Copyright (c) 2018 Kohei Nagasawa
// Read LICENSE.md for license condition of this software
//

#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <functional>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#undef _USE_MATH_DEFINES

// GL includes
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// CUDA includes
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_cuda_gl.h>

// Imgui includes
#ifdef USE_IMGUI
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_glfw.h>
#endif

#include "mandelbulbRenderer.cuh"

// The main application class
class MandelbulbApp
{
public:

    // Settings from users
    struct Settings
    {
        // Windows settings
        int m_windowWidth = 512;
        int m_windowHeight = 512;
        float m_windowAngle = 60.f;
    };

    // Parameters for Mandelbulb calculation
    struct Parameters
    {
        // Mandelbulb parameters
        float m_laplacianThreshold = 0.5f;
        float m_iterationToDeltaStepRate = 0.55f;
        float m_iterationAccelerator = 0.f;
        float m_depthToDeltaStepRate = 0.0071f;
        float m_deltaDepth;
        int m_minimumIterations = 3;
        int m_numDrillingIterations = 5;
        bool m_adaptiveIterationCounts = true;
        bool m_distanceEstimation = false;
        bool m_mortonCurve = true;
        bool m_doublePrecision = false;

        // Rendering settings
        MandelbulbRenderer::NormalMode m_normalMode = MandelbulbRenderer::ScreenSpace;
        MandelbulbRenderer::ColoringMode m_coloringMode = MandelbulbRenderer::Normal;
        bool m_castShadow = false;
        bool m_applySSAO = true;
    };

    // Camera object
    struct Camera
    {
        float3 m_position;
        float3 m_forward;
        float3 m_up;
        float3 m_side;
        float m_speed;
        float m_distFromOrigin;
        float m_focalDepth;
    };

    // Mouse cursor object
    struct Cursor
    {
        enum State
        {
            Drag,
            None,
        };

        float2 m_prevPosition;
        State m_state = None;
    };

public:

    // Destructor
    ~MandelbulbApp() = default;

    // Get the instance
    static MandelbulbApp* getInstance();

    // Kill the instance
    static void quit();

    // Initialization
    bool init(const Settings& settings);

    // Main loop
    void run();

private:

    // Constructor
    MandelbulbApp() = default;

    // Set window size to closest power of 2
    void setWindowSize(int size);

    // Initialization functions for OpenGL
    bool initGL();
    bool initShaders();
    bool initTexture();
    bool initVertexBuffer();
    bool initPixelBuffer();

    // Initialization of CUDA
    bool initCUDA();

    // Initialization of Imgui
    bool initImgui();

    // Render Imgui
    void updateImgui();

    // Termination
    void terminate();

    // Core process of rendering Mandelbulb
    void renderMandelbulb();

    // Handle key inputs
    void handleKeyInputsImpl(int key, int scancode, int action, int mods);
    // Static function as callback because glfw works with only C based functions
    static void handleKeyInputsCallback(GLFWwindow* window, int key, int scancode, int action, int mods);

    // Handle cursor (mouse) inputs
    void handleMouseButtonInputsImpl(int button, int action, int mods);
    void handleCursorInputsImpl(float xpos, float ypos);
    // Static function as callback because glfw works with only C based functions
    static void handleMouseButtonInputsCallback(GLFWwindow* window, int button, int action, int mods);
    static void handleCursorInputsCallback(GLFWwindow* window, double xpos, double ypos);

    // Camera manipulation
    // Update camera status of Mandelbulb renderer
    void updateCamera();
    // Update speed of camera based on the result of rendering Mandelbulb
    void calcCameraSpeed();
    // Move camera along the direction with the current speed
    void moveCamera(const float3 direction);
    // Rotate camera by delta of screen space distances
    void rotateCamera(float xDiff, float yDiff);

    // Calculate frame rate
    void calcFramePerSecond();

    // Calculate distance from the origin
    void calcDistFromOrigin();

private:

    // The window
    GLFWwindow* m_window = nullptr;

    // Array of colors for each pixel
    GLubyte* m_pixelColors = nullptr;

    // Pixel buffer object ID
    GLuint m_pbo = 0;

    // Vertex array object ID
    GLuint m_vao = 0;

    // Vertex buffer object for quad geometry and UV coordinates
    GLuint m_vbos[2] = { 0, 0 };

    // Texture ID
    GLuint m_texture;

    // User settings
    Settings m_settings;

    // Parameters
    Parameters m_params;

    // The camera
    Camera m_camera;

    // The cursor
    Cursor m_cursor;

    // The Mandelbulb renderer
    MandelbulbRenderer* m_mandelbulb = nullptr;

    // The number of pixels
    unsigned int m_numPixels;

    // Frame per second
    float m_millisec = 0.f;
    float m_fps = 0.f;

    bool m_drawDebugMenu = true;

    bool m_isPausing = false;

#ifdef USE_IMGUI
    ImGuiIO* m_imguiIO = nullptr;
#endif

    // Singleton instance
    static MandelbulbApp* s_app;
};

MandelbulbApp* MandelbulbApp::s_app = nullptr;
MandelbulbApp::Settings s_settings;

// Get the singleton instance
MandelbulbApp* MandelbulbApp::getInstance()
{
    if (!s_app)
    {
        s_app = new MandelbulbApp();
        if (!s_app->init(s_settings))
        {
            std::cout << "Failed to initialize application" << std::endl;
            getchar();

            return nullptr;
        }
    }

    return s_app;
}

// Kill the instance
void MandelbulbApp::quit()
{
    if (s_app)
    {
        s_app->terminate();
        delete s_app;
        s_app = nullptr;
    }
}

// Initialization
bool MandelbulbApp::init(const Settings& settings)
{
    atexit(MandelbulbApp::quit);

    // Take settings from input
    m_settings = settings;

    setWindowSize(m_settings.m_windowWidth);

    // Allocate pixel color buffer
    m_numPixels = m_settings.m_windowWidth * m_settings.m_windowHeight;
    m_pixelColors = new GLubyte[3 * m_numPixels];

    // Initialize OpenGL
    if (!initGL())
    {
        std::cout << "Failed to initialize OpenGL" << std::endl;
        return false;
    }

    // Initialize CUDA
    if (!initCUDA())
    {
        std::cout << "Failed to initialize CUDA" << std::endl;
        return false;
    }

    // Initialize imgui
    if (!initImgui())
    {
        std::cout << "Failed to initialize imgui" << std::endl;
    }

    // Create Mandelbulb renderer
    {
        float const cameraAngle = m_settings.m_windowAngle * (float)M_PI / 180.0f;
        float const pixelAngle = cameraAngle / (float)m_settings.m_windowWidth;

        m_mandelbulb = new MandelbulbRenderer(m_settings.m_windowWidth, m_settings.m_windowHeight);
        m_mandelbulb->addLight({ 5.0, -5.0, -3.0 });
        m_mandelbulb->addLight({ -5.0,  5.0,  3.0 });
        m_mandelbulb->setPixelAngle(pixelAngle);

        m_mandelbulb->setMinimumIterations(m_params.m_minimumIterations);
        m_mandelbulb->enableAdaptiveIterations(m_params.m_adaptiveIterationCounts);
        m_mandelbulb->setIterationAccelerator(m_params.m_iterationAccelerator);
        m_mandelbulb->setLaplacianThreshold(m_params.m_laplacianThreshold);
        m_mandelbulb->setIterationToDeltaStepRate(m_params.m_iterationToDeltaStepRate);
        m_mandelbulb->setDepthToDeltaStepRate(m_params.m_depthToDeltaStepRate);
        m_mandelbulb->setNumDrillingIterations(m_params.m_numDrillingIterations);
        m_mandelbulb->enableMortonCurveIndexing(m_params.m_mortonCurve);
        m_mandelbulb->enableDoublePrecision(m_params.m_doublePrecision);
        m_mandelbulb->enableDistanceEstimation(m_params.m_distanceEstimation);

        m_mandelbulb->setNormalMode(m_params.m_normalMode);
    }

    // Initialize the camera
    {
        m_camera.m_forward.x = 0.f;
        m_camera.m_forward.y = 0.f;
        m_camera.m_forward.z = -1.f;
        m_camera.m_up.x = 0.f;
        m_camera.m_up.y = 1.f;
        m_camera.m_up.z = 0.f;
        m_camera.m_side.x = -1.f;
        m_camera.m_side.y = 0.f;
        m_camera.m_side.z = 0.f;
        m_camera.m_position.x = 0.f;
        m_camera.m_position.y = 0.f;
        m_camera.m_position.z = 3.5f;
        m_camera.m_focalDepth = m_mandelbulb->getFocalDepth();
        calcCameraSpeed();
    }

    calcDistFromOrigin();
    m_params.m_deltaDepth = 0.01f;

    m_mandelbulb->setInitialDeltaStep(m_params.m_deltaDepth);

    updateCamera();

    return true;
}

// Main loop
void MandelbulbApp::run()
{
    // Main loop
    // Check if the ESC key was pressed or the window was closed
    while (glfwGetKey(m_window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
        glfwWindowShouldClose(m_window) == 0)
    {
        if (m_isPausing)
        {
            glfwPollEvents();
            continue;
        }

        glClear(GL_COLOR_BUFFER_BIT);

        // Render Mandelbulb!
        renderMandelbulb();

        // Pour the pixels into the texture
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
        glBindTexture(GL_TEXTURE_2D, m_texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_settings.m_windowWidth, m_settings.m_windowHeight, GL_RGB, GL_UNSIGNED_BYTE, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Draw the quad
        glBindVertexArray(m_vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);

        // Update frame rate
        calcFramePerSecond();

        // Draw the debug menu
        if (m_drawDebugMenu)
        {
            updateImgui();
        }

        // Swap buffers
        glfwSwapBuffers(m_window);
        glfwPollEvents();

        // Recalculate camera speed
        calcCameraSpeed();
    }
}

// Set window size to closest power of 2
void MandelbulbApp::setWindowSize(int sizeIn)
{
    auto getExp = [](int value)
    {
        int count = 0;
        int current = value;
        while (current > 1)
        {
            current = current / 2;
            ++count;
        }

        return count;
    };

    // The window size has to be power of 2 and width and height have to be the same.
    // This is because currently Morton curve indexing supports only power of 2 numbers.
    // [todo] Support arbitrary numbers in Morton curve indexing
    int size = (int)std::pow(2.f, (float)getExp(sizeIn));
    m_settings.m_windowWidth = size;
    m_settings.m_windowHeight = size;
}

// Handle key inputs
void MandelbulbApp::handleKeyInputsCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (s_app)
    {
        s_app->handleKeyInputsImpl(key, scancode, action, mods);
    }
}

// Handle key inputs
void MandelbulbApp::handleKeyInputsImpl(int key, int scancode, int action, int mods)
{
#ifdef USE_IMGUI
    if (m_drawDebugMenu)
    {
        if (m_imguiIO->WantCaptureKeyboard)
        {
            ImGui_ImplGlfw_KeyCallback(m_window, key, scancode, action, mods);
            return;
        }
    }
#endif

    switch (key)
    {
    case GLFW_KEY_W:
        moveCamera(m_camera.m_forward);
        break;
    case GLFW_KEY_S:
        moveCamera(-m_camera.m_forward);
        break;
    case GLFW_KEY_A:
        moveCamera(m_camera.m_side);
        break;
    case GLFW_KEY_D:
        moveCamera(-m_camera.m_side);
        break;
    case GLFW_KEY_Q:
        moveCamera(m_camera.m_up);
        break;
    case GLFW_KEY_Z:
        moveCamera(-m_camera.m_up);
        break;
    case GLFW_KEY_R:
        if (action == GLFW_PRESS)
        {
            // Perform ray casting on CPU from the center of the window
            m_mandelbulb->rayMarchOnHost(0.0001f);
        }
        break;
    case GLFW_KEY_M:
        if (action == GLFW_PRESS)
        {
            // Toggle the debug menu
            m_drawDebugMenu = !m_drawDebugMenu;
        }
        break;
    case GLFW_KEY_P:
        if (action == GLFW_PRESS)
        {
            // Toggle pause
            m_isPausing = !m_isPausing;
        }
        break;
    case GLFW_KEY_T:
        if (action == GLFW_PRESS)
        {
            // Output timing information
            m_mandelbulb->profile();
        }
        break;
    }
}

// Handle cursor inputs
void MandelbulbApp::handleMouseButtonInputsCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (s_app)
    {
        s_app->handleMouseButtonInputsImpl(button, action, mods);
    }
}

// Handle cursor inputs
void MandelbulbApp::handleCursorInputsCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (s_app)
    {
        s_app->handleCursorInputsImpl((float) xpos, (float) ypos);
    }
}

// Handle mouse button inputs
void MandelbulbApp::handleMouseButtonInputsImpl(int button, int action, int mods)
{
#ifdef USE_IMGUI
    if (m_drawDebugMenu)
    {
        if (m_imguiIO->WantCaptureMouse)
        {
            ImGui_ImplGlfw_MouseButtonCallback(m_window, button, action, mods);
            return;
        }
    }
#endif

    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if(action == GLFW_PRESS && m_cursor.m_state == Cursor::None)
        {
            m_cursor.m_state = Cursor::Drag;
            return;
        }
        else if (action == GLFW_RELEASE && m_cursor.m_state == Cursor::Drag)
        {
            m_cursor.m_state = Cursor::None;
        }
    }
}

// Handle cursor inputs
void MandelbulbApp::handleCursorInputsImpl(float xpos, float ypos)
{
    if (m_cursor.m_state == Cursor::Drag)
    {
        const float rotationSpeed = 0.002f;
        const float xDiff = xpos - m_cursor.m_prevPosition.x;
        const float yDiff = ypos - m_cursor.m_prevPosition.y;
        rotateCamera(xDiff, yDiff);
    }

    m_cursor.m_prevPosition.x = xpos;
    m_cursor.m_prevPosition.y = ypos;
}

// Core process of rendering Mandelbulb
void MandelbulbApp::renderMandelbulb()
{
    cudaGLMapBufferObject((void**)&m_pixelColors, m_pbo);

    // Main calculation
    m_mandelbulb->createMandelbulb(m_pixelColors);

    m_params.m_deltaDepth = m_mandelbulb->getInitialDeltaStep();
    m_camera.m_focalDepth = m_mandelbulb->getFocalDepth();

    cudaGLUnmapBufferObject(m_pbo);
}

// Update camera status of Mandelbulb renderer
void MandelbulbApp::updateCamera()
{
    m_mandelbulb->updateCamera(m_camera.m_position, m_camera.m_forward, m_camera.m_up, m_camera.m_side);
}

// Update speed of camera based on the result of rendering Mandelbulb
void MandelbulbApp::calcCameraSpeed()
{
    m_camera.m_speed = m_camera.m_focalDepth * 0.03f;
}

// Move camera along the direction with the current speed
void MandelbulbApp::moveCamera(const float3 direction)
{
    m_camera.m_position += direction * m_camera.m_speed;
    calcDistFromOrigin();
    updateCamera();
}

// Rotate camera by delta of screen space distances
void MandelbulbApp::rotateCamera(float xDiff, float yDiff)
{
    auto normalize = [](const float3 a)
    {
        float length = sqrt(dot(a, a));
        return length > 0 ? a / length : a;
    };

    auto rotateAxis = [](const float3 vector, const float3 axis, const float angle)
    {
        float3 vectorOut;
        const float c = cos(angle);
        const float s = sin(angle);
        const float oneMinusC = 1 - c;
        const float cosx = oneMinusC * axis.x;
        const float cosy = oneMinusC * axis.y;
        const float cosz = oneMinusC * axis.z;
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
    };

    const float rotationSpeed = 0.001f;
    xDiff *= -rotationSpeed;
    yDiff *= rotationSpeed;

    m_camera.m_forward = rotateAxis(m_camera.m_forward, m_camera.m_up, xDiff);
    m_camera.m_side    = normalize(rotateAxis(m_camera.m_side, m_camera.m_up, xDiff));
    m_camera.m_forward = normalize(rotateAxis(m_camera.m_forward, m_camera.m_side, yDiff));
    m_camera.m_up      = normalize(rotateAxis(m_camera.m_up, m_camera.m_side, yDiff));

    updateCamera();
}

// Calculate frame rate
void MandelbulbApp::calcFramePerSecond()
{
    static clock_t clock1, clock2;
    static int count = 0;
    if (count == 0)
    {
        clock1 = clock();
    }

    ++count;

    // Update frame rate every 5 frames
    if (count >= 5)
    {
        clock2 = clock();
        const clock_t deltaClock = clock2 - clock1;
        m_fps = (float)CLOCKS_PER_SEC / deltaClock * count;
        m_millisec = 1000.f / m_fps;
        count = 0;
    }
}

// Calculate distance from the origin
void MandelbulbApp::calcDistFromOrigin()
{
    m_camera.m_distFromOrigin = sqrtf(dot(m_camera.m_position, m_camera.m_position));
}

// Termination
void MandelbulbApp::terminate()
{
    if (m_mandelbulb)
    {
        delete m_mandelbulb;
        m_mandelbulb = nullptr;
    }

    if (m_pbo)
    {
        cudaGLUnregisterBufferObject(m_pbo);
        glDeleteBuffers(1, &m_pbo);
    }

    if (m_vao)
    {
        glDeleteVertexArrays(1, &m_vao);
    }

    for (int i = 0; i < 2; ++i)
    {
        if (m_vbos[i])
        {
            glDeleteBuffers(1, &m_vbos[i]);
        }
    }

    if (m_texture)
    {
        glDeleteTextures(1, &m_texture);
    }

    // Close OpenGL window and terminate GLFW
    glfwTerminate();

#ifdef USE_IMGUI
    // Terminate imgui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
#endif
}

// Initialize OpenGL
bool MandelbulbApp::initGL()
{
    // Initialize GLFW
    if (!glfwInit())
    {
        std::cout << "Failed to initialize GLFW\n" << std::endl;
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Open a window and create its OpenGL context
    m_window = glfwCreateWindow(m_settings.m_windowWidth, m_settings.m_windowHeight, "Mandelbulb", nullptr, nullptr);
    if (m_window == nullptr)
    {
        std::cout << "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials." << std::endl;
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(m_window);

    // Initialize GLEW
    if (glewInit() != GLEW_OK)
    {
        std::cout << "Failed to initialize GLEW" << std::endl;
        glfwTerminate();
        return false;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(m_window, GLFW_STICKY_KEYS, GL_TRUE);

    // Set background color
    glClearColor(1.0f, 1.0f, 1.0f, 0.0f);

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);

    if (!initShaders())
    {
        std::cout << "Failed to initialize shaders" << std::endl;
        return false;
    }

    if (!initTexture())
    {
        std::cout << "Failed to initialize texture" << std::endl;
        return false;
    }

    if (!initVertexBuffer())
    {
        std::cout << "Failed to initialize vertex buffers" << std::endl;
        return false;
    }

    if (!initPixelBuffer())
    {
        std::cout << "Failed to initialize pixel buffer object" << std::endl;
        return false;
    }

    // Register key input callback
    glfwSetKeyCallback(m_window, MandelbulbApp::handleKeyInputsCallback);

    // Register mouse input callback
    glfwSetMouseButtonCallback(m_window, MandelbulbApp::handleMouseButtonInputsCallback);
    glfwSetCursorPosCallback(m_window, MandelbulbApp::handleCursorInputsCallback);

    return true;
}

// Initialize shaders
bool MandelbulbApp::initShaders()
{
    // Print shader compiler errors
    auto printShaderInfoLog = [this](GLuint shader, const char *fileName)
    {
        std::cerr << "Compiler Error in" << fileName << std::endl;

        // Get length of error log from compiler
        GLsizei bufSize;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &bufSize);
        if (bufSize > 1)
        {
            // Get the error log
            std::vector<GLchar> log(bufSize);
            GLsizei length;
            glGetShaderInfoLog(shader, bufSize, &length, &log[0]);
            std::cerr << &log[0] << std::endl;
        }
    };

    // Print shader linker errors
    auto printProgramInfoLog = [this](GLuint program)
    {
        std::cerr << "Linker Error" << std::endl;

        // Get length of error log from linker
        GLsizei bufSize;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &bufSize);
        if (bufSize > 1)
        {
            // Get the error log
            std::vector<GLchar> log(bufSize);
            GLsizei length;
            glGetProgramInfoLog(program, bufSize, &length, &log[0]);
            std::cerr << &log[0] << std::endl;
        }
    };

    // Read shader source from file
    auto readShaderSource = [this](const char *name, std::vector<GLchar> &buffer)
    {
        if (name == NULL)
        {
            return false;
        }

        std::ifstream file(name, std::ios::binary);
        if (file.fail())
        {
            std::cerr << "Error: Failed open source file: " << name << std::endl;
            return false;
        }

        file.seekg(0L, std::ios::end);

        {
            const GLsizei length = static_cast<GLsizei>(file.tellg());
            buffer.resize(length + 1);
            file.seekg(0L, std::ios::beg);
            file.read(buffer.data(), length);
            buffer[length] = '\0';
        }

        if (file.fail())
        {
            std::cerr << "Error: Failed to read source file: " << name << std::endl;
        }

        file.close();
        return true;
    };

    // Create a shading program
    const GLuint program(glCreateProgram());

    // Read shaders from files
    const char* vertexShaderName = "point.vert";
    const char* fragmentShaderName = "texture.frag";

    std::vector<GLchar> vertexShader, fragmentShader;
    if (!readShaderSource(vertexShaderName, vertexShader))
    {
        return false;
    }

    if (!readShaderSource(fragmentShaderName, fragmentShader))
    {
        return false;
    }

    auto compileShader = [&program, printShaderInfoLog](const char* fileName, const char* src, GLenum type)
    {
        const GLuint obj(glCreateShader(type));
        glShaderSource(obj, 1, &src, NULL);
        glCompileShader(obj);

        // Get result of compiling the shader
        GLint status;
        glGetShaderiv(obj, GL_COMPILE_STATUS, &status);
        if (status == GL_FALSE)
        {
            printShaderInfoLog(obj, fileName);
            glDeleteShader(obj);
            return false;
        }

        glAttachShader(program, obj);

        return true;
    };

    // Compile shaders
    if (!compileShader(vertexShaderName, vertexShader.data(), GL_VERTEX_SHADER))
    {
        return false;
    }

    if (!compileShader(fragmentShaderName, fragmentShader.data(), GL_FRAGMENT_SHADER))
    {
        return false;
    }

    // Link Shaders
    {
        glLinkProgram(program);

        // Get result of linking shaders
        GLint status;
        glGetProgramiv(program, GL_LINK_STATUS, &status);
        if (status == GL_FALSE)
        {
            printProgramInfoLog(program);
        }
    }

    glUseProgram(program);

    return true;
}

// Initialize texture
bool MandelbulbApp::initTexture()
{
    // Create texture
    glActiveTexture(GL_TEXTURE0);
    glGenTextures(1, &m_texture);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, m_settings.m_windowWidth, m_settings.m_windowHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    return true;
}

// Initialize vertex array and vertex buffer objects
bool MandelbulbApp::initVertexBuffer()
{
    // Quad
    constexpr GLfloat vertices[] =
    {
        -1.f,  1.f,
        -1.f, -1.f,
        1.f, -1.f,
        1.f, -1.f,
        1.f,  1.f,
        -1.f,  1.f,
    };

    // UV
    constexpr GLfloat texcoords[] =
    {
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f,
    };

    // Create vertex arrays
    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    // Create vertex buffer object

    // Create vertex buffer for quad
    glGenBuffers(2, m_vbos);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbos[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    // Create vertex buffer for UV
    glBindBuffer(GL_ARRAY_BUFFER, m_vbos[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return true;
}

// Initialize pixel buffer object
bool MandelbulbApp::initPixelBuffer()
{
    // Create pixel buffer object
    int numValues = m_numPixels * 4;
    int texureDataSize = sizeof(GLubyte) * numValues;

    // Generate a buffer
    glGenBuffers(1, &m_pbo);
    // Make this the current UNPACK buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    // Allocate data for the buffer
    glBufferData(GL_PIXEL_UNPACK_BUFFER, texureDataSize, NULL, GL_DYNAMIC_COPY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    return true;
}

// Initialize CUDA
bool MandelbulbApp::initCUDA()
{
    cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

    // Register pixel buffer object
    cudaGLRegisterBufferObject(m_pbo);

    return true;
}

#ifdef USE_IMGUI

// Initialize Imgui
bool MandelbulbApp::initImgui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    m_imguiIO = &ImGui::GetIO();

    ImGui_ImplGlfw_InitForOpenGL(m_window, false);
    // Register imgui's callbacks for scroll and text input because they are not used in the main app
    glfwSetScrollCallback(m_window, ImGui_ImplGlfw_ScrollCallback);
    glfwSetCharCallback(m_window, ImGui_ImplGlfw_CharCallback);

    const char* glsl_version = "#version 130";
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImGui::StyleColorsDark();

    return true;
}

// Update Imgui
void MandelbulbApp::updateImgui()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    {
        static float f = 0.0f;
        static int counter = 0;

        ImGui::Begin("Configuration");

        // FPS
        ImGui::Text("%.3f ms/frame (%.1f FPS)", m_millisec, m_fps);

        // Camera info and control
        if (ImGui::CollapsingHeader("Camera"))
        {
            ImGui::InputFloat3("Position", &m_camera.m_position.x);
            ImGui::InputFloat3("Direction", &m_camera.m_forward.x);
            // [todo] Add button to jump to several places
            ImGui::Text("Focal Depth = %f", m_camera.m_focalDepth);
            ImGui::Text("Distance from Origin = %f", m_camera.m_distFromOrigin);
            ImGui::Text("View Angle = %f", m_settings.m_windowAngle);
        }

        // Parameters to calculate Mandelbulb
        if (ImGui::CollapsingHeader("Mandelbulb"))
        {
            ImGui::InputInt("Minimum Iterations", &m_params.m_minimumIterations);
            if (ImGui::IsItemDeactivatedAfterEdit())
            {
                m_mandelbulb->setMinimumIterations(m_params.m_minimumIterations);
            }
            ImGui::Checkbox("Adaptive Iteration Counts", &m_params.m_adaptiveIterationCounts);
            if (ImGui::IsItemEdited())
            {
                m_mandelbulb->enableAdaptiveIterations(m_params.m_adaptiveIterationCounts);
            }
            if (m_params.m_adaptiveIterationCounts)
            {
                ImGui::SliderFloat("Laplacian Threshold", &m_params.m_laplacianThreshold, 0.f, 1.f);
                if (ImGui::IsItemDeactivatedAfterEdit())
                {
                    m_mandelbulb->setLaplacianThreshold(m_params.m_laplacianThreshold);
                }

                ImGui::SliderFloat("Iteration to Depth Rate", &m_params.m_iterationToDeltaStepRate, 0.f, 1.f);
                if (ImGui::IsItemDeactivatedAfterEdit())
                {
                    m_mandelbulb->setIterationToDeltaStepRate(m_params.m_iterationToDeltaStepRate);
                }

                ImGui::InputInt("Drill Iterations", &m_params.m_numDrillingIterations);
                if (ImGui::IsItemDeactivatedAfterEdit())
                {
                    m_mandelbulb->setNumDrillingIterations(m_params.m_numDrillingIterations);
                }
            }
            ImGui::Checkbox("Distance Estimation", &m_params.m_distanceEstimation);
            if (ImGui::IsItemEdited())
            {
                m_mandelbulb->enableDistanceEstimation(m_params.m_distanceEstimation);
            }
            ImGui::SliderFloat("Iteration Accelerator", &m_params.m_iterationAccelerator, 0.f, 5.f);
            if (ImGui::IsItemDeactivatedAfterEdit())
            {
                m_mandelbulb->setIterationAccelerator(m_params.m_iterationAccelerator);
            }
            ImGui::SliderFloat("Depth to Delta Step Rate", &m_params.m_depthToDeltaStepRate, 0.000001f, 0.02f);
            if (ImGui::IsItemDeactivatedAfterEdit())
            {
                m_mandelbulb->setDepthToDeltaStepRate(m_params.m_depthToDeltaStepRate);
            }

            ImGui::Checkbox("Morton Curve", &m_params.m_mortonCurve);
            m_mandelbulb->enableMortonCurveIndexing(m_params.m_mortonCurve);

            ImGui::Checkbox("Double Precision", &m_params.m_doublePrecision);
            m_mandelbulb->enableDoublePrecision(m_params.m_doublePrecision);
        }

        // Rendering settings
        if (ImGui::CollapsingHeader("Rendering"))
        {
            {
                const char* listbox_items[] = { "No Normal", "Pseudo SS Normal", "Screen Space Normal" };
                ImGui::Combo("Normal Type", (int*)&m_params.m_normalMode, listbox_items, IM_ARRAYSIZE(listbox_items));
                m_mandelbulb->setNormalMode(m_params.m_normalMode);
            }

            ImGui::Checkbox("Shadow", &m_params.m_castShadow);
            m_mandelbulb->enableShadow(m_params.m_castShadow);

            ImGui::Checkbox("SSAO", &m_params.m_applySSAO);
            m_mandelbulb->enableSSAO(m_params.m_applySSAO);

            {
                const char* listbox_items[] = { "Basic", "Colorful" };
                ImGui::Combo("Coloring Mode", (int*)&m_params.m_coloringMode, listbox_items, IM_ARRAYSIZE(listbox_items));
                m_mandelbulb->setColoringMode(m_params.m_coloringMode);
            }
        }

        ImGui::End();
    }

    ImGui::Render();
    glfwMakeContextCurrent(m_window);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

#else

bool MandelbulbApp::initImgui()
{
    std::cout << "Imgui was not built with this application." << std::endl;
    std::cout << "You cannot modify parameters of Mandelbulb renderer without Imgui." << std::endl;
    std::cout << "For the full experience of this application, please download Imgui and build the app again with it." << std::endl;
    return true;
}

void MandelbulbApp::updateImgui()
{
}

#endif

// Handle given command arguments
bool handleArgumentsParameters(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if ((arg == "-w") || (arg == "--width"))
        {
            if (i + 1 < argc)
            {
                s_settings.m_windowWidth = atoi(argv[++i]);
            }
            else
            {
                std::cerr << "--width option requires one argument." << std::endl;
                return false;
            }
        }
        if ((arg == "-h") || (arg == "--height"))
        {
            if (i + 1 < argc)
            {
                s_settings.m_windowHeight = atoi(argv[++i]);
            }
            else
            {
                std::cerr << "--height option requires one argument." << std::endl;
                return false;
            }
        }
        if ((arg == "-a") || (arg == "--angle"))
        {
            if (i + 1 < argc)
            {
                s_settings.m_windowAngle = (float)atof(argv[++i]);
            }
            else
            {
                std::cerr << "--angle option requires one argument." << std::endl;
                return false;
            }
        }
    }

    return true;
}

// Entry point
int main(int argc, char* argv[])
{
    // Take input from command line
    if (argc > 1 && std::string(argv[1]) == "help")
    {
        std::cout << "Usage: $0 [--option <argument>]..." << std::endl;
        std::cout << std::endl;
        std::cout << "-w --width     Width of window" << std::endl;
        std::cout << "-h --height    Height of window" << std::endl;
        std::cout << "-a --angle     View angle of camera" << std::endl;
        return 0;
    }

    if (!handleArgumentsParameters(argc, argv))
    {
        return 1;
    }

    MandelbulbApp* app = MandelbulbApp::getInstance();
    if (app)
    {
        app->run();
    }

    return 0;
}
