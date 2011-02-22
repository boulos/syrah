#ifndef _SYRAH_ATOMIC_COUNTER_H_
#define _SYRAH_ATOMIC_COUNTER_H_

#ifdef WIN32
#include "Mutex.h"
#endif

#include "Preprocessor.h"

namespace syrah {

/**************************************

 CLASS
 AtomicCounter

 KEYWORDS
 Thread

 DESCRIPTION
 Provides a simple atomic counter.  This will work just like an
 integer, but guarantees atomicty of the ++ and -- operators.
 Despite their convenience, you do not want to make a large number
 of these objects.  See also WorkQueue.

 Not that this implementation does not offer an operator=, but
 instead uses a "set" method.  This is to avoid the inadvertant
 use of a statement like: x=x+2, which would NOT be thread safe.

****************************************/
class AtomicCounter {
public:
  AtomicCounter(int value = 0);
  ~AtomicCounter();

  //////////
  // Allows the atomic counter to be used in expressions like
  // a normal integer.  Note that multiple calls to this function
  // may return different values if other threads are manipulating
  // the counter.
  operator int() const;

  //////////
  // Increment the counter and return the new value.
  // This does not return AtomicCounter& like a normal ++
  // operator would, because it would destroy atomicity
  int operator++();

  //////////
  //    Increment the counter and return the old value
  int operator++(int);

  // Increment the counter by the given value and return the old
  // value. Like operator++ this doesn't return AtomicCounter& to
  // avoid atomicity issues.
  int operator+=(int);

  //////////
  // Decrement the counter and return the new value
  // This does not return AtomicCounter& like a normal --
  // operator would, because it would destroy atomicity
  int operator--();

  //////////
  // Decrement the counter and return the old value
  int operator--(int);

  //////////
  // Set the counter to a new value
  void set(int);

private:
  int value;
  // some padding to avoid false sharing between counters.
  char padding[128 - sizeof(int)];

#ifdef WIN32
  Mutex valueLock;
#endif

  // Cannot copy them
  AtomicCounter(const AtomicCounter&);
  AtomicCounter& operator=(const AtomicCounter&);
};

/*
 *  AtomicCounter: Thread-safe integer variable written in x86 assembly
 *
 *  Written by:
 *   Author: Solomon Boulos
 *   Department of Computer Science
 *   University of Utah
 *   Date: 10-Sep-2007
 *
 */
SYRAH_FORCEINLINE AtomicCounter::AtomicCounter(int value) : value(value) {
}

SYRAH_FORCEINLINE AtomicCounter::~AtomicCounter() {
}

SYRAH_FORCEINLINE AtomicCounter::operator int() const {
  return value;
}

SYRAH_FORCEINLINE
int
AtomicCounter::operator++() {
#ifdef WIN32
  valueLock.lock();
  int rval = ++value;
  valueLock.unlock();
  return rval;
#else
  __volatile__ register int return_val = 1;

  __asm__ __volatile__(
      "lock;\n"
      "xaddl %1, %0;\n" :
      "+m" (value), "+r"(return_val) :
      "m" (value) , "r" (return_val)
      /* no unknown clobbers */
    );
  return return_val + 1;
#endif
}

SYRAH_FORCEINLINE
int
AtomicCounter::operator++(int) {
#ifdef WIN32
  valueLock.lock();
  int rval = value++;
  valueLock.unlock();
  return rval;
#else
  __volatile__ register int return_val = 1;

  __asm__ __volatile__(
      "lock;\n"
      "xaddl %1, %0;\n" :
      "+m" (value), "+r"(return_val) :
      "m" (value) , "r" (return_val)
      /* no unknown clobbers */
    );
  return return_val;
#endif
}

SYRAH_FORCEINLINE
int
AtomicCounter::operator+=(int val) {
#ifdef WIN32
  valueLock.lock();
  int rval = value;
  value += val;
  valueLock.unlock();
  return rval;
#else
  __volatile__ register int return_val = val;

  __asm__ __volatile__(
      "lock;\n"
      "xaddl %1, %0;\n" :
      "+m" (value), "+r"(return_val) :
      "m" (value) , "r" (return_val), "r" (val)
      /* no unknown clobbers */
    );
  return return_val;
#endif
}

SYRAH_FORCEINLINE
int
AtomicCounter::operator--() {
#ifdef WIN32
  valueLock.lock();
  int rval = --value;
  valueLock.unlock();
  return rval;
#else
  __volatile__ register int return_val = -1;
  __asm__ __volatile__(
      "lock;\n"
      "xaddl %1, %0;\n" :
      "+m" (value), "+r"(return_val) :
      "m" (value) , "r" (return_val)
      /* no unknown clobbers */
    );
  return return_val - 1;
#endif
}

SYRAH_FORCEINLINE
int
AtomicCounter::operator--(int) {
#ifdef WIN32
  valueLock.lock();
  int rval = value++;
  valueLock.unlock();
  return rval;
#else
  __volatile__ register int return_val = -1;
  __asm__ __volatile__(
      "lock;\n"
      "xaddl %1, %0;\n" :
      "+m" (value), "+r"(return_val) :
      "m" (value) , "r" (return_val)
      /* no unknown clobbers */
    );
  // The exchange returns the old value
  return return_val;
#endif
}

SYRAH_FORCEINLINE
void
AtomicCounter::set(int v) {
#ifdef WIN32
  valueLock.lock();
  value = v;
  valueLock.unlock();
#else
  __volatile__ register int copy_val = v;
  __asm__ __volatile__(
    "lock;\n"
    "xchgl %1, %0\n" :
    "+m" (value), "+r" (copy_val) :
    "m" (value), "r" (copy_val), "r" (v)
    /* no unknown clobbers */
    );
#endif
}

} // end namespace syrah

#endif // _SYRAH_ATOMIC_COUNTER_H_

