#include <stdio.h>
#include <unistd.h>

#include "syrah/CycleTimer.h"

using namespace syrah;

#define SLEEP_TESTS                                                 \
   SLEEP_TEST(UsleepShort, usleep, 8371, CycleTimer::usPerTick)     \
   SLEEP_TEST(UsleepLong,  usleep, 127381, CycleTimer::usPerTick)   \
   SLEEP_TEST(SleepShort, sleep, 2, CycleTimer::secondsPerTick)     \
   SLEEP_TEST(SleepLong, sleep, 5, CycleTimer::secondsPerTick)      \


bool CloseEnough(unsigned expected, double actual) {
   double ratio = actual / expected;
   return (ratio > .9 && ratio < 1.1);
}

#define SLEEP_TEST(name, sleepFunc, amount, conv)      \
   static bool SleepTest_##name() {                    \
      CycleTimer::SysClock elapsed;                   \
      double actual;                                  \
      elapsed = CycleTimer::currentTicks();           \
      sleepFunc(amount);                              \
      elapsed = CycleTimer::currentTicks() - elapsed; \
      actual = conv() * elapsed;                      \
      return CloseEnough(amount, actual);             \
   }

SLEEP_TESTS
#undef SLEEP_TEST

int main() {
   int numFailed = 0;
#define SLEEP_TEST(name, sleepFunc, amount, conv)       \
   if (!SleepTest_##name()) {                           \
      numFailed++;                                      \
      fprintf(stderr, "%s test failed.\n", #name);      \
   }

   SLEEP_TESTS;
#undef SLEEP_TEST

   if (numFailed) {
      fprintf(stderr, "%d tests failed.\n", numFailed);
      return -1;
   }

   fprintf(stderr, "All tests passed.\n");
   return 0;
}
