#ifndef SYRAH_MUTEX_POOL_H_
#define SYRAH_MUTEX_POOL_H_

#include "Mutex.h"

namespace syrah {
  class MutexPool {
  public:
    MutexPool(int how_many) : mutexes(new Mutex[how_many]) {
    }

    ~MutexPool() {
      delete[] mutexes;
    }

    void lock(int which) {
      mutexes[which].lock();
    }

    void unlock(int which) {
      mutexes[which].unlock();
    }

    bool tryLock(int which) {
      return mutexes[which].tryLock();
    }

  private:
    Mutex* mutexes;
  };
};

#endif // SYRAH_MUTEX_POOL_H_
