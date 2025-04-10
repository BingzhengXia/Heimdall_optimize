#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.57])
AC_INIT([HEIMDALL], [0.1], [ajameson@swin.edu.au])

AC_CONFIG_SRCDIR([Applications/heimdall.C])
AC_CONFIG_MACRO_DIR([config])

AC_ARG_VAR([PSRHOME],    [Standard pulsar home directory])
AC_ARG_VAR([LOGIN_ARCH], [Architecture-dependent sub-directory of PSRHOME])

AC_PREFIX_DEFAULT([${PSRHOME:-"/usr/local"}${PSRHOME:+"/$LOGIN_ARCH"}])

AM_INIT_AUTOMAKE([foreign subdir-objects])
AC_DISABLE_SHARED

AC_CANONICAL_HOST
case $host in
  i?86-*-linux-*) ARCH=IA32;;
  ia64-*-linux-*) ARCH=IA64;;
  x86_64-*-linux-*) ARCH=AMD64;;
  *) ARCH=unsupported;
     AC_MSG_WARN([architecture is not supported by third party software]);;
esac
AC_SUBST(ARCH)

# Checks for programs.
AC_PROG_CC
AC_PROG_CPP
AC_PROG_CXX
AM_PROG_CC_C_O
AC_PROG_INSTALL
AC_PROG_LIBTOOL

# 添加C++17支持
CXXFLAGS="$CXXFLAGS -std=c++17"

LIBS="$PTHREAD_LIBS $LIBS"
CFLAGS="$CFLAGS $PTHREAD_CFLAGS"
CXXFLAGS="$CXXFLAGS $PTHREAD_CFLAGS"

SWIN_LIB_DEDISP
SWIN_LIB_CUDA
SWIN_LIB_PSRDADA
SWIN_LIB_THRUST
BOOST_REQUIRE([1.4])

# Checks for header files.
AC_HEADER_STDC
AC_CHECK_HEADERS([arpa/inet.h fcntl.h inttypes.h netdb.h netinet/in.h stdlib.h string.h sys/socket.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE
AC_TYPE_INT64_T
AC_TYPE_SIZE_T
AC_TYPE_UINT64_T
AC_C_CONST
AC_TYPE_PID_T
AC_TYPE_SIZE_T
AC_HEADER_TIME
AC_STRUCT_TM

# Checks for library functions.
AC_FUNC_FORK
AC_FUNC_MALLOC
AC_FUNC_MEMCMP
AC_FUNC_MKTIME
AC_FUNC_REALLOC
AC_FUNC_SELECT_ARGTYPES
AC_FUNC_SETVBUF_REVERSED
AC_TYPE_SIGNAL
AC_FUNC_STAT
AC_FUNC_STRFTIME
AC_FUNC_VPRINTF
AC_FUNC_ERROR_AT_LINE
AC_CHECK_FUNCS([gettimeofday memset pow socket sqrt strdup strerror])


if test "x$prefix" = xNONE; then
  prefix=$ac_default_prefix
fi

HEIMDALL_INSTALL=$prefix
AC_SUBST(HEIMDALL_INSTALL)

AM_CONFIG_HEADER([config.h])

AC_CONFIG_FILES([Applications/Makefile
                 Formats/Makefile
                 Makefile
                 Network/Makefile
                 Pipeline/Makefile
                 Share/Makefile
                 Scripts/Makefile
                 Scripts/trans_paths.py])
AC_OUTPUT
