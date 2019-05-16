#ifndef __GUIDDEF__
#define __GUIDDEF__ 1

typedef struct _GUID {
    unsigned int  Data1;   // unsigned long  64/32 bit diff in linux
    unsigned short Data2;
    unsigned short Data3;
    unsigned char  Data4[8];
} GUID;

typedef GUID IID;
typedef IID *LPIID;

#ifndef EXTERN_C
#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C extern
#endif
#endif

#ifdef DEFINE_GUID
#undef DEFINE_GUID
#endif

#ifdef INITGUID
#define DEFINE_GUID(name, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) \
        EXTERN_C const GUID name \
                = { l, w1, w2, { b1, b2,  b3,  b4,  b5,  b6,  b7,  b8 } }
#else
#define DEFINE_GUID(name, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8) \
    EXTERN_C const GUID name
#endif // INITGUID


#define REFGUID const GUID &
                
__inline int IsEqualGUID(REFGUID rguid1, REFGUID rguid2)
{
   return (
      ((unsigned int *) &rguid1)[0] == ((unsigned int *) &rguid2)[0] &&
      ((unsigned int *) &rguid1)[1] == ((unsigned int *) &rguid2)[1] &&
      ((unsigned int *) &rguid1)[2] == ((unsigned int *) &rguid2)[2] &&
      ((unsigned int *) &rguid1)[3] == ((unsigned int *) &rguid2)[3]);
}


#ifdef __cplusplus
__inline int operator==(REFGUID guidOne, REFGUID guidOther)
{
    return IsEqualGUID(guidOne,guidOther);
}

__inline int operator!=(REFGUID guidOne, REFGUID guidOther)
{
    return !(guidOne == guidOther);
}
#endif

#endif
