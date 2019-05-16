#ifndef __STUBS__
#define __STUBS__ 1

#define LCOMP_LINUX 1

#ifndef LCOMP_LINUX
#include <windows.h>
#define AO_t int
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <atomic_ops.h>
#endif


typedef struct { volatile AO_t counter; } atomic_t;

// windows types...
#ifdef LCOMP_LINUX

#include "guiddef.h"

#define __stdcall


typedef void *LPVOID;
typedef void *PVOID;
typedef int   HANDLE;
typedef void *HMODULE;
typedef void *HINSTANCE;
typedef char CHAR, *PCHAR;
typedef unsigned char UCHAR, *PUCHAR;
typedef unsigned int ULONG, ULONG32, *PULONG;
typedef int LONG, *PLONG;
typedef short SHORT, *PSHORT;
typedef unsigned short USHORT, *PUSHORT;
typedef int BOOL;

typedef LONG HRESULT;

typedef struct _OVERLAPPED {
    ULONG Internal;
    ULONG InternalHigh;
    union {
        struct {
            ULONG Offset;
            ULONG OffsetHigh;
        };

        PVOID Pointer;
    };

    HANDLE  hEvent;
} OVERLAPPED, *LPOVERLAPPED;


#define TRUE 1
#define FALSE 0
#define INVALID_HANDLE_VALUE ((HANDLE)(-1))


#define ERROR_INVALID_FUNCTION           1L    // dderror
#define ERROR_FILE_NOT_FOUND             2L
#define ERROR_PATH_NOT_FOUND             3L
#define ERROR_TOO_MANY_OPEN_FILES        4L
#define ERROR_ACCESS_DENIED              5L
#define ERROR_INVALID_HANDLE             6L

#ifdef RC_INVOKED
#define _HRESULT_TYPEDEF_(_sc) _sc
#else // RC_INVOKED
#define _HRESULT_TYPEDEF_(_sc) ((HRESULT)_sc)
#endif // RC_INVOKED

#define E_NOINTERFACE                    _HRESULT_TYPEDEF_(0x80004002L)
#define S_OK                             ((HRESULT)0x00000000L)


#ifndef NULL
#ifdef __cplusplus
#define NULL    0
#else
#define NULL    ((void *)0)
#endif
#endif



// some math function
#define l_fabs(x) ((x>=0) ? x:(-x))
#define l_ceil(x) ((double)((int)x+1))

#ifdef CONFIG_SMP
#define LOCK_PREFIX \
      ".section .smp_locks,\"a\"\n" \
      "  .align 4\n"       \
      "  .long 661f\n" /* address */   \
      ".previous\n"        \
            "661:\n\tlock; "

#else /* ! CONFIG_SMP */
#define LOCK_PREFIX ""
#endif

#endif


BOOL LFreeLibrary(HINSTANCE handle);
HINSTANCE LLoadLibrary(const char *szLibFileName);
void *LGetProcAddress(HINSTANCE handle, const char *szProcName);


BOOL LCloseHandle(HANDLE hDevice);
HANDLE LCreateFile(const char *szDrvName);

BOOL LDeviceIoControl(HANDLE hDevice,
                       ULONG dwIoControlCode,
                       LPVOID lpInBuffer,
                       ULONG nInBufferSize,
                       LPVOID lpOutBuffer,
                       ULONG nOutBufferSize,
                       PULONG lpBytesReturned,
                       LPOVERLAPPED lpOverlapped);

//=====================================
// this for ARM with helper
//typedef int (__kuser_cmpxchg_t)(int oldval, int newval, volatile int *ptr);
//#define __kuser_cmpxchg (*(__kuser_cmpxchg_t *)0xffff0fc0)
//
//static int atomic_add(volatile int *ptr, int val)
// {
//        int old, _new;
//
//        do {
//                old = *ptr;
//                _new = old + val;
//        } while(__kuser_cmpxchg(old, _new, ptr));
//
//        return _new;
//}

static /*__inline__*/ void atomic_inc(atomic_t *v)
{
#ifdef LCOMP_LINUX
// this for x86 x64 on asm
//   __asm__ __volatile__(
//      LOCK_PREFIX "incl %0"
//      :"+m" (v->counter));
//===============================

//    atomic_add(&v->counter,1); // arm with helper
    AO_fetch_and_add1(&v->counter);
#else
   InterlockedIncrement((LONG *)&(v->counter));
#endif
}

static /*__inline__*/ void atomic_dec(atomic_t *v)
{
#ifdef LCOMP_LINUX
// this for x86 x64 on asm
//   __asm__ __volatile__(
//      LOCK_PREFIX "decl %0"
//      :"+m" (v->counter));
// =============================

//    atomic_add(&v->counter,-1); // arm with helper
    AO_fetch_and_sub1(&v->counter);
#else
   InterlockedDecrement((LONG *)&(v->counter));
#endif
}

#endif

