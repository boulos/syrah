#ifndef _SYRAH_PREPROCESSOR_H_
#define _SYRAH_PREPROCESSOR_H_

#if defined (__INTEL_COMPILER)
#define SYRAH_ALIGN(size) \
  __declspec(align((size)))
#elif defined (__GNUC__)
#define SYRAH_ALIGN(size) \
  __attribute__ ((aligned (size)))
#elif defined (_MSC_VER)
#define SYRAH_ALIGN(size) \
  __declspec(align(size))
#else
#define SYRAH_ALIGN(size)
#endif

#define SYRAH_IS_ALIGNED(addr,size) \
  ((((size_t)(addr))%(size))==0)

#define SYRAH_IS_ALIGNED_POW2(addr,size) \
  ((((size_t)(addr))&((size)-1))==0)


#if defined(__GNUC__)
#define SYRAH_FUNC __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define SYRAH_FUNC __FUNCTION__ // or perhaps __FUNCDNAME__ + __FUNCSIG__
#else
#define SYRAH_FUNC __func__
#endif

// NOTE(boulos): ISO C99 defines _Pragma to let you do this.
#define SYRAH_PRAGMA(str) _Pragma (#str)

// Even with the Intel Compiler and Qstd=c99, this won't work on
// Win32...
#if defined(__INTEL_COMPILER) && !defined(_WIN32)
//#define SYRAH_UNROLL(amt) SYRAH_PRAGMA(unroll((amt)))
#define SYRAH_UNROLL(amt)
#else
// NOTE(boulos): Assuming GCC
#define SYRAH_UNROLL(amt)
#endif

#if defined(__INTEL_COMPILER)
#define SYRAH_FORCEINLINE __forceinline
#elif defined(_MSC_VER)
#define SYRAH_FORCEINLINE inline
#else
// NOTE(boulos): Assuming GCC
#define SYRAH_FORCEINLINE __attribute__ ((always_inline)) inline
#endif

// Stack allocation equivalent to type[size]
#define SYRAH_STACK_ALLOC(type, size) ((type*)alloca((size) * sizeof(type)))

#endif // _SYRAH_PREPROCESSOR_H_
