#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

#include "syrah/Barrier.h"
#include "syrah/CycleTimer.h"
#include "syrah/Task.h"
#include "syrah/Thread.h"
#include "syrah/Mutex.h"
using namespace syrah;

class SaxpyTask : public Task {
public:
  SaxpyTask(float a, float x, float y) : a(a), x(x), y(y) {
  }

  virtual void Run() {
    result = a * x + y;
  }

  float a;
  float x;
  float y;
  float result;
};

static int FibonacciNumber(int n) {
  if (n < 2) return 1;
  return FibonacciNumber(n - 2) + FibonacciNumber(n - 1);
}


class FibonacciTask : public Task {
public:
  FibonacciTask(int a) : a(a) {
  }


  virtual void Run() {
    result = FibonacciNumber(a);
  }

  int a;
  int result;
};

void* ConsumeTasks(TaskQueue* queue) {
  while (Task* cur_task = queue->pop()) {
    cur_task->Launch();
  }
  return 0;
}

class ConsumeTaskThreadStart : public ThreadStart {
public:
  virtual void* Run(void* args) {
    return ConsumeTasks((TaskQueue*)args);
  }
};

// NOTE(boulos): Interesting. This is consistently slower for the
// Fibonnacci test usually running 3.89 seconds vs 3.52 ish. The
// difference is not just the overhead of starting up the threads,
// since the SaxpyTest doesn't demonstrate the same difference.
void RunTest(const int kNumThreads, TaskQueue* task_queue) {
  ThreadStart* start = new ConsumeTaskThreadStart;
  std::vector<Thread*> threads;
  ThreadDesc desc;
  desc.start = start;
  desc.stack_size = 1024 * 1024 * 1;
  desc.args = (void*)task_queue;
  for (int i = 0; i < kNumThreads; i++) {
    Thread* thread = new Thread(desc);
    threads.push_back(thread);
  }
  for (int i = 0; i < kNumThreads; i++) {
    threads[i]->Start();
  }
  for (int i = 0; i < kNumThreads; i++) {
    threads[i]->BlockUntilFinished();
  }
  for (int i = 0; i < kNumThreads; i++) {
    delete threads[i];
  }
}

void SaxpyTest(const int kNumThreads, const int kNumTasks) {
  TaskList saxpy_tasks;
  for (int i = 0; i < kNumTasks; i++) {
    saxpy_tasks.push_back(new SaxpyTask(drand48(), drand48(), drand48()));
  }

  TaskQueue task_queue;
  task_queue.push_back(&saxpy_tasks);
  double start = CycleTimer::currentSeconds();
  RunTest(kNumThreads, &task_queue);
  double end = CycleTimer::currentSeconds();
  double total_time = end - start;
  double tasks_per_second = kNumTasks / total_time;
  fprintf(stderr, "Finished %d saxpy tasks in %.2f seconds (%.2f tasks per second)\n", kNumTasks, total_time, tasks_per_second);
}

void FibonacciTest(const int kNumThreads, const int kNumTasks) {
  TaskList fib_tasks;
  for (int i = 0; i < kNumTasks; i++) {
    fib_tasks.push_back(new FibonacciTask(1 + static_cast<int>(drand48() * 20)));
  }
  TaskQueue task_queue;
  task_queue.push_back(&fib_tasks);
  double start = CycleTimer::currentSeconds();
  RunTest(kNumThreads, &task_queue);
  double end = CycleTimer::currentSeconds();
  double total_time = end - start;
  double tasks_per_second = kNumTasks / total_time;

  fprintf(stderr, "Finished %d fibonacci tasks in %.2f seconds (%.2f tasks per second)\n", kNumTasks, total_time, tasks_per_second);
}

class MutexTestThreadStart : public ThreadStart {
public:
  MutexTestThreadStart() : result(0) {
  }

  struct TestArgs {
    int num_increments;
  };
  virtual void* Run(void* args) {
    TestArgs* test_args = (TestArgs*)args;
    for (int i = 0; i < test_args->num_increments; i++) {
      mutex.lock();
      result++;
      mutex.unlock();
    }
    return NULL;
  }
  Mutex mutex;
  int result;
};

void MutexIncrementTest(const int kNumThreads, const int kIncrementsPerThread) {
  std::vector<Thread*> threads;
  MutexTestThreadStart* test = new MutexTestThreadStart();
  MutexTestThreadStart::TestArgs args;
  args.num_increments = kIncrementsPerThread;
  ThreadDesc desc;
  desc.start = test;
  desc.stack_size = 1024 * 1024 * 1;
  desc.args = &args;

  for (int i = 0; i < kNumThreads; i++) {
    threads.push_back(new Thread(desc));
  }
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i]->Start();
  }
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i]->BlockUntilFinished();
  }
  for (size_t i = 0; i < threads.size(); i++) {
    delete threads[i];
  }
  fprintf(stderr, "Mutex test got result %d\n", test->result);
  delete test;
}

class BarrierTestThreadStart : public ThreadStart {
public:
  BarrierTestThreadStart() : data(NULL), result(0) {
  }

  struct TestArgs {
    int num_threads;
  };
  virtual void* Run(void* args) {
    TestArgs* test_args = (TestArgs*)args;
    int num_threads = test_args->num_threads;
    int which = barrier.wait(num_threads);
    if (which == 0) {
      data = new int[num_threads];
    }
    int second_which = barrier.wait(num_threads);
    data[second_which] = FibonacciNumber(which);
    which = barrier.wait(num_threads);
    if (which == 0) {
      result = 0;
      for (int i = 0; i < num_threads; i++) {
        printf("data[%d] = %d\n", i, data[i]);
        result += data[i];
      }
      delete data;
    }
    return NULL;
  }
  Barrier barrier;
  int* data;
  int result;
};

void BarrierTest(const int kNumThreads) {
  std::vector<Thread*> threads;
  BarrierTestThreadStart* test = new BarrierTestThreadStart();
  BarrierTestThreadStart::TestArgs args;
  args.num_threads = kNumThreads;
  ThreadDesc desc;
  desc.start = test;
  desc.stack_size = 1024 * 1024 * 1;
  desc.args = &args;

  for (int i = 0; i < kNumThreads; i++) {
    threads.push_back(new Thread(desc));
  }
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i]->Start();
  }
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i]->BlockUntilFinished();
  }
  for (size_t i = 0; i < threads.size(); i++) {
    delete threads[i];
  }
  fprintf(stderr, "Barrier test got result %d\n", test->result);
  delete test;
}



int main(int argc, char** argv) {
  int kNumThreads = 2;
  int kMegaTasks =  1;
  if (argc != 1) {
    kNumThreads = atoi(argv[1]);
    kMegaTasks = atoi(argv[2]);
  }
  SaxpyTest(kNumThreads, 1024 * 1024 * kMegaTasks);
  FibonacciTest(kNumThreads, 1024 * 1024 * kMegaTasks);
  MutexIncrementTest(kNumThreads, (1024 * 1024 * kMegaTasks) / kNumThreads);
  BarrierTest(kNumThreads);

  return 0;
}
