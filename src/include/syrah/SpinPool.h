#ifndef SYRAH_SPIN_POOL_H_
#define SYRAH_SPIN_POOL_H_

#if defined(WIN32)
#include "MutexPool.h"
namespace syrah {
  // NOTE(boulos): lamely you can't do typedef syrah::MutexPool syrah::SpinPool...
  class SpinPool {
  public:
    SpinPool(int how_many) : pool(how_many) {
    }

    SYRAH_FORCEINLINE void lock(int which) {
      pool.lock(which);
    }

    SYRAH_FORCEINLINE void unlock(int which) {
      pool.unlock(which);
    }

    SYRAH_FORCEINLINE bool tryLock(int which) {
      return pool.tryLock(which);
    }
  private:
    MutexPool pool;
  };
}
#else

#include "SpinLock.h"

namespace syrah {
  class SpinPool {
  private:
    struct SYRAH_ALIGN(128) PaddedLock {
      SYRAH_ALIGN(128) SpinLock actual_lock;

      SYRAH_FORCEINLINE void lock() {
        actual_lock.lock();
      }
      SYRAH_FORCEINLINE void unlock() {
        actual_lock.unlock();
      }
      SYRAH_FORCEINLINE bool tryLock() {
        return actual_lock.tryLock();
      }
    };
  public:
    SpinPool(int how_many) : locks(new PaddedLock[how_many]) {
    }

    ~SpinPool() {
      delete[] locks;
    }

    SYRAH_FORCEINLINE void lock(int which) {
      locks[which].lock();
    }

    SYRAH_FORCEINLINE void unlock(int which) {
      locks[which].unlock();
    }

    SYRAH_FORCEINLINE bool tryLock(int which) {
      return locks[which].tryLock();
    }

  private:
    PaddedLock* locks;
  };
};
#endif

#endif // SYRAH_MUTEX_POOL_H_
