ADD_LIBRARY(cuda_comp SHARED kernel.h kernel.cu)

TARGET_COMPILE_FEATURES(cuda_comp PRIVATE cuda_std_17)

SET_TARGET_PROPERTIES(cuda_comp
        PROPERTIES
        CUDA_RUNTIME_LIBRARY Shared
        CUDA_SEPARABLE_COMPILATION ON # i have no idea what this does
)

ADD_EXECUTABLE(main main.cpp)

TARGET_COMPILE_FEATURES(main PRIVATE cxx_std_17)
# set_target_properties(main CXX_STANDARD 17)

TARGET_LINK_LIBRARIES(main cuda_comp)