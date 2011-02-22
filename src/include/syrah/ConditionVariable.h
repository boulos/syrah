#ifndef _SYRAH_CONDITION_VARIABLE_H_
#define _SYRAH_CONDITION_VARIABLE_H_

#include "Mutex.h"

namespace syrah {
  class ConditionVariable {
  public:
    ConditionVariable() {
      pthread_cond_init(&condition, NULL);
    }

    void wait(Mutex& m) {
      pthread_cond_wait(&condition, (pthread_mutex_t*)(m.opaquePtr()));
    }

    // unblock at least one thread
    void wakeOne() {
      pthread_cond_signal(&condition);
    }

    void wakeAll() {
      pthread_cond_broadcast(&condition);
    }

  private:
    pthread_cond_t condition;
  };
};

#endif // _SYRAH_CONDITION_VARIABLE_H_
