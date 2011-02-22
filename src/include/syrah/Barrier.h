#ifndef _SYRAH_BARRIER_H_
#define _SYRAH_BARRIER_H_

#include "ConditionVariable.h"
#include "Mutex.h"

namespace syrah {
  class Barrier {
  public:
    Barrier() : num_waiting(0), sense(false) {
    }

    int wait(int numThreads) {
      wait_mutex.lock();
      int which = ++num_waiting; // get the new value by using pre-increment
      ConditionVariable& cond = (sense) ? cond0 : cond1;

      if (which == numThreads) {
        // I was the last one, reset the num_waiting, change the sense
        // and then wake people up.
        num_waiting = 0;
        sense = !sense;
        cond.wakeAll();
      } else {
        cond.wait(wait_mutex);
      }
      wait_mutex.unlock();
      return which - 1;
    }

  private:
    // Don't copy me
    Barrier(const Barrier&);
    Barrier& operator=(const Barrier&);

    // NOTE(boulos): Using an atomic counter here doesn't make much
    // sense since we want to hold the mutex immediately afterwards
    // anyway...
    int num_waiting;
    Mutex wait_mutex;
    bool sense;
    ConditionVariable cond0;
    ConditionVariable cond1;
  };
};

#endif // _SYRAH_BARRIER_H_
