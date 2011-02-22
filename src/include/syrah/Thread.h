#ifndef _SYRAH_THREAD_H_
#define _SYRAH_THREAD_H_

#include <pthread.h>

namespace syrah {
  // Probably want the same Callback mechanism Manta provides. Some
  // static args and some dynamic args. Especially since forcing the
  // pthread_create "syntax" on people can be lame.
  class ThreadStart {
  public:
    virtual ~ThreadStart() {}
    virtual void* Run(void* args) = 0;
  };

  struct ThreadDesc {
    ThreadStart* start;
    size_t stack_size;
    void* args;
    // TODO(boulos): Allow affinity/locality hints (i.e. L2-thread
    // placement on Leopard, exact core placement on Linux, etc)
  };

  class Thread {
  public:
    // TODO(boulos): Should threads require no args for construction
    // and then have Set* functions? This would allow the creation of
    // an array of threads more easily. Then again, it may be better
    // to simply expose the idea of a ThreadGroup. Especially if
    // Threads can be added/deleted on demand.
    Thread(const ThreadDesc& desc) : desc(desc) {
    }

    ~Thread() {
    }

    // TODO(boulos): Allow barrier'ed start (create all threads and
    // then block until all started). This probably works better in a
    // ThreadGroup setting (though the semantics of joining the group
    // half way through a wave are unclear and should be well defined).
    void Start() {
      int result = pthread_create(&thread, NULL, Thread::StartThreads, this);
      // TODO(boulos): set attribute stack size
      if (result != 0) throw "Failed to create thread";
    }

    void BlockUntilFinished() {
      pthread_join(thread, NULL);
    }

    // TODO(boulos): These all seem like good functions to have.

    //void Stop();
    //void Pause();
    //void Resume();

    // This is so I can just pass Thread* to StartThreads. I'm open to
    // other ideas.
    const ThreadDesc* getDesc() const {
      return &desc;
    }
  private:
    static void* StartThreads(void* args) {
      Thread* thread = (Thread*)args;
      const ThreadDesc* desc = thread->getDesc();
      return desc->start->Run(desc->args);
    }

    // Disallow copy constructors
    Thread(const Thread&);
    Thread& operator=(const Thread&);

    pthread_t thread;
    ThreadDesc desc;
  };
};

#endif // _SYRAH_THREAD_H_
