#ifndef _SYRAH_TASK_H_
#define _SYRAH_TASK_H_

#include "AtomicCounter.h"
#include "SpinLock.h"
#include <vector>
#include <cassert>

namespace syrah {

  class Task;

  class TaskList {
  public:
    TaskList() : num_assigned(0), num_remaining(0) {
    }

    ~TaskList();

    void push_back(Task* task);
    // pop isn't really right
    Task* pop() {
      unsigned int which = num_assigned++;
      return tasks[which];
    }

    Task* const& operator[](int i) const {
      return tasks[i];
    }



    size_t size() const {
      return tasks.size();
    }
    void markAsFinished(unsigned int task_id) {
      num_remaining--;
    }
    bool Done() const {
      return (num_remaining == 0);
    }

  private:
    std::vector<Task*> tasks;
    // This conflates the scheduling with the object
    AtomicCounter num_assigned;
    AtomicCounter num_remaining;
  };

  class Task {
  public:
    virtual ~Task() {}
    virtual void Run() = 0;
    unsigned int GetId() const { return task_id; }

    void Launch() {
      Run();
      parent_list->markAsFinished(task_id);
    }
    friend class TaskList;

  protected:
    void AddToTaskList(TaskList* parent, unsigned int id) {
      parent_list = parent;
      task_id = id;
    }

  private:

    unsigned int task_id;
    TaskList* parent_list;
  };

  class TaskQueue {
  public:
    TaskQueue() {
      task_lists.resize(8, NULL);
      read_index = write_index = 0;
    }
    // Maybe this should be push as dependency of parent?
    void push_back(TaskList* new_work) {
      queue_mutex.lock();
      task_lists[write_index] = new_work;
      write_index++;
      size_t cur_size = task_lists.size();
      // Assume cur_size is pow2, then write_index % cur_size is
      // below.
      write_index &= (cur_size - 1);
      if (write_index == read_index) {
        // Just hit the circular buffer loop point, resize the queue
        task_lists.resize(2 * cur_size, NULL);
        // Set write_index to end of old circular buffer and
        // read_index to 0 (as we have a full set in [0, cur_size)
        // already).
        write_index = cur_size;
        read_index = 0;
      }
      queue_mutex.unlock();
    }

    // Grab a Task from the queue. Returns NULL if no work to do.
    Task* pop() {
      Task* result = 0;
      queue_mutex.lock();
      // queue is only empty or full if read_index == write_index
      if (read_index != write_index) {
        TaskList* top_list = task_lists[read_index];
        result = top_list->pop();
        unsigned int task_idx = result->GetId();
        if (task_idx + 1 == top_list->size()) {
          // popped the last task from the list, remove it from the
          // "work to do" queue
          read_index++;
          read_index &= (task_lists.size() - 1);
        }
      }
      queue_mutex.unlock();
      return result;
    }
  private:
    std::vector<TaskList*> task_lists;
    size_t read_index;
    size_t write_index;

    SpinLock queue_mutex;
  };

  // Have to wait until Task is fully declared.
  inline TaskList::~TaskList() {
    for (size_t i = 0; i < tasks.size(); i++) {
      delete tasks[i];
    }
  }

  inline void TaskList::push_back(Task* task) {
    assert(num_assigned == 0);
    task->AddToTaskList(this, tasks.size());
    // XXX(boulos): It is an error to add tasks while the TaskList is
    // running.
    tasks.push_back(task);
    // TODO(boulos): Don't do this. Just set it equal to size once the
    // thing is inserted.
    num_remaining++;
  }

};


#endif // _SYRAH_TASK_H_
