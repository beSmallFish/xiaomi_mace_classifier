# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../app/libs/${ANDROID_ABI})

#OpenCV-android-sdk路径
set(pathToOpenCv D:/android_demo/tensorFlow/OpenCV-android-sdk)
#配置加载native依赖->引入OpenCV头文件
include_directories(${pathToOpenCv}/sdk/native/jni/include)
#引入libopencv_java3.so文件
add_library(opencv_java3 SHARED IMPORTED)
set_target_properties(opencv_java3
                        PROPERTIES IMPORTED_LOCATION
                       ${CMAKE_SOURCE_DIR}/src/main/cpp/lib/arm64-v8a/libopencv_java3.so)

include_directories(${CMAKE_SOURCE_DIR}/)
include_directories(${CMAKE_SOURCE_DIR}/src/main/cpp/include)
set(mace_lib ${CMAKE_SOURCE_DIR}/src/main/cpp/lib/arm64-v8a/libmace.a)
set(mobilenet_lib ${CMAKE_SOURCE_DIR}/src/main/cpp/model/arm64-v8a/wangsu.a)
add_library (mace_lib STATIC IMPORTED)
set_target_properties(mace_lib PROPERTIES IMPORTED_LOCATION ${mace_lib})
add_library (mobilenet_lib STATIC IMPORTED)
set_target_properties(mobilenet_lib PROPERTIES IMPORTED_LOCATION ${mobilenet_lib})

add_library( # Sets the name of the library.
             mace_mobile_jni

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             src/main/cpp/opencv/thread_event.cc
             src/main/cpp/opencv/native-lib.cpp
             src/main/cpp/image_classify.cc )

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
                       mace_mobile_jni
                       mace_lib
                       mobilenet_lib
                       opencv_java3
                       jnigraphics

                       # Links the target library to the log library
                       # included in the NDK.
                       ${log-lib} )