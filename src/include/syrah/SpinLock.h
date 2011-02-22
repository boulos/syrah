#ifndef SYRAH_SPINLOCK_H_
#define SYRAH_SPINLOCK_H_

#include "Preprocessor.h"

#define SYRAH_USE_APPLE_SPINLOCK 0

#if SYRAH_USE_APPLE_SPINLOCK
#include <libkern/OSAtomic.h>
#endif

namespace syrah {
  class SpinLock {
  public:
    SpinLock() : value(0) {
    }
    int value;

    SYRAH_FORCEINLINE void unlock() {
#if SYRAH_SYRAH_USE_APPLE_SPINLOCK
      OSSpinLockUnlock((OSSpinLock*)&value);
#else
      volatile int return_val;
      do {
        return_val = 0;
        __asm__ __volatile__(
          "lock;\n"
          "xchg %1, %0;\n" :
          "+m" (value), "+r"(return_val) :
          "m" (value) , "r" (return_val)
          /* no unknown clobbers */
          );
        // Check that we unlocked the lock (meaning that return_val = 1)
      } while (return_val == 0);

      return;
#endif
    }


    SYRAH_FORCEINLINE void lock() {
#if SYRAH_USE_APPLE_SPINLOCK
      OSSpinLockLock((OSSpinLock*)&value);
#else
      volatile int return_val;

      do {
        return_val = 1;
        __asm__ __volatile__(
          "lock;\n"
          "xchg %1, %0;\n" :
          "+m" (value), "+r"(return_val) :
          "m" (value) , "r" (return_val)
          /* no unknown clobbers */
          );
        // Check that we got the lock, so return_val == 0 to exit
      } while (return_val == 1);
      return;
#endif
    }

    SYRAH_FORCEINLINE bool tryLock() {
#if SYRAH_USE_APPLE_SPINLOCK
      return OSSpinLockTry((OSSpinLock*)&value);
#else
      volatile int return_val = 1;

      __asm__ __volatile__(
        "lock;\n"
        "xchg %1, %0;\n" :
        "+m" (value), "+r"(return_val) :
        "m" (value) , "r" (return_val)
        /* no unknown clobbers */
        );
      return return_val == 0;
#endif
    }
  };
}

#endif // SYRAH_SPINLOCK_H_
