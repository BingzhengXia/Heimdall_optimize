
include $(top_srcdir)/cuda.mk

lib_LTLIBRARIES = libhdpipeline.la

libhdpipeline_la_SOURCES = default_params.C error.C parse_command_line.C clean_filterbank_rfi.cu get_rms.cu matched_filter.cu remove_baseline.cu find_giants.cu label_candidate_clusters.cu merge_candidates.cu pipeline.cu measure_bandpass.cu median_filter.cu matched_filter.cu 

nobase_include_HEADERS = hd/median_filter.h hd/error.h hd/types.h

# To find the dedisp lib from bbarsdell
AM_CPPFLAGS = -I$(top_srcdir) \
	      -I$(top_srcdir)/Network \
	      -I$(top_srcdir)/Formats \
	      @DEDISP_CFLAGS@ @THRUST_CFLAGS@ @CUDA_CFLAGS@ $(BOOST_CPPFLAGS)

include $(top_srcdir)/config/Makefile.targets

CUDA_NVCC_FLAGS = --default-stream per-thread -Xcompiler "-fopenmp" -O2 -arch=compute_70 -code=sm_70 # 为 CUDA 编译器添加 OpenMP 支持
CXXFLAGS = -g -O2 -fopenmp  # 为 C++ 编译器添加 OpenMP 支持
CFLAGS = -g -O2 -fopenmp  # 为 C 编译器添加 OpenMP 支持
