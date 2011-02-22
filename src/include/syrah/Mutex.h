#ifndef _SYRAH_MUTEX_H_
#define _SYRAH_MUTEX_H_

namespace syrah {
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
  class Mutex {
  public:
    Mutex() {
      mutex = CreateMutex(NULL, 0, 0);
      if (mutex == 0) throw "Failed to initialize mutex";
    }

    ~Mutex() {
      CloseHandle(mutex);
    }

    void lock() {
      WaitForSingleObject(mutex, INFINITE);
    }

    void unlock() {
      ReleaseMutex(mutex);
    }

    bool tryLock() {
      int result = WaitForSingleObject(mutex, 0);
      if (result == WAIT_OBJECT_0) return true;
      return false;
    }
  private:
    HANDLE mutex;
  };
#else
  class Mutex {
  public:
    Mutex() {
      if (pthread_mutex_init(&mutex, NULL) != 0) throw "Failed to initialize mutex";
    }

    ~Mutex() {
      pthread_mutex_destroy(&mutex);
    }
    void lock() {
      pthread_mutex_lock(&mutex);
    }

    void unlock() {
      pthread_mutex_unlock(&mutex);
    }

    bool tryLock() {
      int result = pthread_mutex_trylock(&mutex);
      return result == 0;
    }

    // NOTE(boulos): For ConditionVariable (needs raw mutex pointer)
    void* opaquePtr() { return (void*)&mutex; }

  private:
    pthread_mutex_t mutex;
  };
#endif
} // end namespace syrah

#endif // _SYRAH_MUTEX_H_
