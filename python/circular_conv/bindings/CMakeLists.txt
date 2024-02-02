# Copyright 2020 Free Software Foundation, Inc.
#
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

########################################################################
# Check if there is C++ code at all
########################################################################
if(NOT circular_conv_sources)
    message(STATUS "No C++ sources... skipping python bindings")
    return()
endif(NOT circular_conv_sources)

########################################################################
# Check for pygccxml
########################################################################
gr_python_check_module_raw("pygccxml" "import pygccxml" PYGCCXML_FOUND)

include(GrPybind)

########################################################################
# Python Bindings
########################################################################

list(APPEND circular_conv_python_files
    circular_conv_python.cc python_bindings.cc)

gr_pybind_make_oot(circular_conv ../../.. gr::circular_conv "${circular_conv_python_files}")

# copy bindings extension for use in QA test module
add_custom_command(
    TARGET circular_conv_python
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:circular_conv_python>
            ${PROJECT_BINARY_DIR}/test_modules/gnuradio/circular_conv/)

install(
    TARGETS circular_conv_python
    DESTINATION ${GR_PYTHON_DIR}/gnuradio/circular_conv
    COMPONENT pythonapi)
