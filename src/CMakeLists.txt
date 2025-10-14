ADD_LIBRARY(kernel SHARED kernel.h kernel.cu)

TARGET_COMPILE_FEATURES(kernel PRIVATE cuda_std_17)

SET_TARGET_PROPERTIES(
        kernel
        PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared
        CUDA_SEPARABLE_COMPILATION ON # i have no idea what this does
)

ADD_EXECUTABLE(main main.cpp)

TARGET_COMPILE_FEATURES(main PRIVATE cxx_std_17)  # cxx_std_17
# set_target_properties(main CXX_STANDARD 17)

TARGET_LINK_LIBRARIES(main kernel CUDA::curand)
