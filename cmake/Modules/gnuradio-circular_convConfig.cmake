find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_CIRCULAR_CONV gnuradio-circular_conv)

FIND_PATH(
    GR_CIRCULAR_CONV_INCLUDE_DIRS
    NAMES gnuradio/circular_conv/api.h
    HINTS $ENV{CIRCULAR_CONV_DIR}/include
        ${PC_CIRCULAR_CONV_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_CIRCULAR_CONV_LIBRARIES
    NAMES gnuradio-circular_conv
    HINTS $ENV{CIRCULAR_CONV_DIR}/lib
        ${PC_CIRCULAR_CONV_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-circular_convTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_CIRCULAR_CONV DEFAULT_MSG GR_CIRCULAR_CONV_LIBRARIES GR_CIRCULAR_CONV_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_CIRCULAR_CONV_LIBRARIES GR_CIRCULAR_CONV_INCLUDE_DIRS)
