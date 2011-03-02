#include "syrah/FixedVector.h"

#include <cstdio>
#include <iostream>
#include <iomanip>
#include <float.h>
#include <cmath>
#include <typeinfo>

using namespace syrah;

bool syrah_debug_verbose = false;
bool exit_on_failure = false;

template <typename T>
const char* TypeToString() {
  if (typeid(T) == typeid(float)) {
    return "float";
  } else if (typeid(T) == typeid(double)) {
    return "double";
  } else if (typeid(T) == typeid(int)) {
    return "int";
  } else if (typeid(T) == typeid(uint8_t)) {
    return "uint8";
  }
  return "UNKNOWN";
}

template<typename T>
struct ValWrapper {
  ValWrapper(T _val) : val(_val) {}
  T val;
};

template<typename T>
inline std::ostream& operator<<(std::ostream& os, const ValWrapper<T>& x) {
  if (typeid(T) == typeid(uint8_t)) {
    os << (int)x.val;
    return os;
  } else {
    os << x.val;
    return os;
  }
}

template <int N>
const char* LengthToString() {
  switch (N) {
  case 1: return "1";
  case 2: return "2";
  case 4: return "4";
  case 8: return "8";
  case 16: return "16";
  case 32: return "32";
  case 64: return "64";
  default: return "UNKNOWN LENGTH";
  }
  return "UNREACHABLE";
}


enum BinaryOp {
  ADD,
  SUB,
  MUL,
  DIV,
  MIN,
  MAX,
};

enum BinaryFloatOp {
  POWF,
  POWI,
  ATAN2
};

enum BitwiseOp {
  AND,
  ANDEQ,
  OR,
  OREQ,
  XOR,
  XOREQ,
  SHL,
  SHLEQ,
  SHR,
  SHREQ
};

enum ComparisonOp {
  LT,
  LE,
  EQ,
  GE,
  GT
};

enum UnaryOp {
  NEG,
  ABS,
  FLOOR,
  CEIL,
  FRAC,
  SIN,
  COS,
  TAN,
  COT,
  EXP,
  LN,
  ATAN,
};

enum TernaryOp {
  MADD
};

enum ReductionOp {
  ReduceMin,
  ReduceMax,
  ReduceSum,
  ReduceProduct
};

enum PermuteOp {
  Reverse
};

static const char* BinaryOpToString(const BinaryOp op) {
  switch (op) {
  case ADD:
    return "ADD";
  case SUB:
    return "SUB";
  case MUL:
    return "MUL";
  case DIV:
    return "DIV";
  case MIN:
    return "MIN";
  case MAX:
    return "MAX";
  }
  return "UNREACHABLE";
}

static const char* BinaryFloatOpToString(const BinaryFloatOp op) {
  switch (op) {
  case POWF:
    return "POWF";
  case POWI:
    return "POWI";
  case ATAN2:
    return "ATAN2";
  }
  return "UNREACHABLE";
}

static const char* BitwiseOpToString(const BitwiseOp op) {
  switch (op) {
  case AND: return "AND";
  case ANDEQ: return "ANDEQ";
  case OR: return "OR";
  case OREQ: return "OREQ";
  case XOR: return "XOR";
  case XOREQ: return "XOREQ";
  case SHL: return "SHL";
  case SHLEQ: return "SHLEQ";
  case SHR: return "SHR";
  case SHREQ: return "SHREQ";
  }
  return "UNREACHABLE";
}

static const char* ComparisonOpToString(const ComparisonOp op) {
  switch (op) {
  case LT:
    return "LT";
  case LE:
    return "LE";
  case EQ:
    return "EQ";
  case GE:
    return "GE";
  case GT:
    return "GT";
  }
  return "UNREACHABLE";
}

static const char* UnaryOpToString(const UnaryOp op) {
  switch (op) {
  case NEG:
    return "NEG";
  case ABS:
    return "ABS";
  case FLOOR:
    return "FLOOR";
  case CEIL:
    return "CEIL";
  case FRAC:
    return "FRAC";
  case SIN:
    return "SIN";
  case COS:
    return "COS";
  case TAN:
    return "TAN";
  case COT:
    return "COT";
  case EXP:
    return "EXP";
  case LN:
    return "LN";
  case ATAN:
    return "ATAN";
  }
  return "UNREACHABLE";
}

static const char* TernaryOpToString(const TernaryOp op) {
  switch (op) {
  case MADD:
    return "MADD";
  }
  return "UNREACHABLE";
}

static const char* ReductionOpToString(const ReductionOp op) {
  switch (op) {
  case ReduceMin:
    return "REDUCE_MIN";
  case ReduceMax:
    return "REDUCE_MAX";
  case ReduceSum:
    return "REDUCE_SUM";
  case ReduceProduct:
    return "REDUCE_PRODUCT";
  }
  return "UNREACHABLE";
}

static const char* PermuteOpToString(const PermuteOp op) {
  switch (op) {
  case Reverse:
    return "REVERSE";
  }
  return "UNREACHABLE";
}

template <typename T>
bool ValueEquals(const T val, const T expected) {
  T epsilon(0);
  if (typeid(T) == typeid(float)) epsilon = 1e-7f;
  if (typeid(T) == typeid(double)) epsilon = 1e-15;

  if (val == expected) return true;
  // Check for NaN (which returns false for the above)
  if (val != val && expected != expected) return true;
  // if -epsilon <= (val - expected) <= epsilon return true
  if (val <= (expected + epsilon) &&
      val >= (expected - epsilon))
    return true;
  return false;
}


#define VEC_STRING "vec_" << LengthToString<N>() << "_" << TypeToString<T>() << "[" << i << "] = " << ValWrapper<T>(vec[i])
#define VECMASK_STRING "vec_" << LengthToString<N>() << "_mask(" << TypeToString<T>() << ")[" << i << "] = " << vec.get(i)


template <typename T, int N>
bool VecEqualsConstant(const FixedVector<T, N>& vec, const T expected_val) {
  int num_failed = 0;
  for (int i = 0; i < N; i++) {
    if (!(ValueEquals<T>(vec[i], expected_val))) {
      std::cerr << VEC_STRING << ". Expected " << ValWrapper<T>(expected_val) << std::endl;
      num_failed++;
      if (exit_on_failure) return false;
    }
  }
  return num_failed == 0;
}

template <typename T, int N>
bool VecEqualsArray(const FixedVector<T, N>& vec, const T* expected_values) {
  int num_failed = 0;
  for (int i = 0; i < N; i++) {
    if (!(ValueEquals<T>(vec[i], expected_values[i]))) {
      std::cerr << VEC_STRING << ". Expected " << ValWrapper<T>(expected_values[i]) << std::endl;
      num_failed++;
      if (exit_on_failure) return false;
    }
  }
  return num_failed == 0;
}

template <typename T, int N>
bool VecEqualsArrayBinaryOp(const FixedVector<T, N>& vec, const T* a, const T* b, BinaryOp op) {
  int num_failed = 0;
  for (int i = 0; i < N; i++) {
    T expected_value = T(0);
    switch (op) {
    case ADD:
      expected_value = a[i] + b[i];
      break;
    case SUB:
      expected_value = a[i] - b[i];
      break;
    case MUL:
      expected_value = a[i] * b[i];
      break;
    case DIV:
      expected_value = a[i] / b[i];
      break;
    case MIN:
      expected_value = std::min(a[i], b[i]);
      break;
    case MAX:
      expected_value = std::max(a[i], b[i]);
      break;
    }
    if (!ValueEquals<T>(vec[i], expected_value)) {
      std::cerr << VEC_STRING << ". Expected " << ValWrapper<T>(expected_value) << " op = " << BinaryOpToString(op) << std::endl;
      num_failed++;
      if (exit_on_failure) return false;
    } else {
      if (syrah_debug_verbose) {
        std::cerr << VEC_STRING << " = " << ValWrapper<T>(a[i]) << " " << BinaryOpToString(op) << " " << ValWrapper<T>(b[i]) << std::endl;
      }
    }
  }
  return num_failed == 0;
}

template <typename T, int N>
bool VecEqualsArrayBinaryFloatOp(const FixedVector<T, N>& vec, const T* a, const T* b, BinaryFloatOp op) {
  int num_failed = 0;
  for (int i = 0; i < N; i++) {
    T expected_value = T(0);
    switch (op) {
    case POWF:
      expected_value = std::pow(a[i], b[i]);
      break;
    case POWI:
      expected_value = std::pow(a[i], b[i]);
      break;
    case ATAN2:
      expected_value = std::atan2(a[i], b[i]);
      break;
    }
    if (!ValueEquals<T>(vec[i], expected_value)) {
      std::cerr << VEC_STRING << ". Expected " << ValWrapper<T>(expected_value) << " op = " << BinaryFloatOpToString(op) << std::endl;
      num_failed++;
      if (exit_on_failure) return false;
    } else {
      if (syrah_debug_verbose) {
        std::cerr << VEC_STRING << " = " << ValWrapper<T>(a[i]) << " " << BinaryFloatOpToString(op) << " " << ValWrapper<T>(b[i]) << std::endl;
      }
    }
  }
  return num_failed == 0;
}

template <typename T, int N>
bool VecEqualsArrayBitwiseOp(const FixedVector<T, N>& vec, const T* a, const T* b, BitwiseOp op) {
  int num_failed = 0;
  for (int i = 0; i < N; i++) {
    T expected_value = T(0);
    switch (op) {
    case AND: expected_value = a[i] & b[i]; break;
    case ANDEQ: expected_value = a[i]; expected_value &= b[i]; break;
    case OR: expected_value = a[i] | b[i]; break;
    case OREQ: expected_value = a[i]; expected_value |= b[i]; break;
    case XOR: expected_value = a[i] ^ b[i]; break;
    case XOREQ: expected_value = a[i]; expected_value ^= b[i]; break;
    case SHL: expected_value = a[i] << b[i]; break;
    case SHLEQ: expected_value = a[i]; expected_value <<= b[i]; break;
    case SHR: expected_value = a[i] >> b[i]; break;
    case SHREQ: expected_value = a[i]; expected_value >>= b[i]; break;
    }
    if (!ValueEquals<T>(vec[i], expected_value)) {
      std::cerr << VEC_STRING << ". Expected " << ValWrapper<T>(expected_value) << " op = " << BitwiseOpToString(op) << std::endl;
      num_failed++;
      if (exit_on_failure) return false;
    } else {
      if (syrah_debug_verbose) {
        std::cerr << VEC_STRING << " = " << ValWrapper<T>(a[i]) << " " << BitwiseOpToString(op) << " " << ValWrapper<T>(b[i]) << std::endl;
      }
    }
  }
  return num_failed == 0;
}


template <typename T, int N>
bool VecEqualsArrayComparisonOp(const FixedVectorMask<N>& vec, const T* a, const T* b, ComparisonOp op) {
  int num_failed = 0;
  for (int i = 0; i < N; i++) {
    bool expected_value = false;
    switch (op) {
    case LT:
      expected_value = (a[i] < b[i]);
      break;
    case LE:
      expected_value = (a[i] <= b[i]);
      break;
    case EQ:
      expected_value = (a[i] == b[i]);
      break;
    case GE:
      expected_value = (a[i] >= b[i]);
      break;
    case GT:
      expected_value = (a[i] > b[i]);
      break;
    }
    if (vec.get(i) != expected_value) {
      std::cerr << VECMASK_STRING << ". Expected " << ValWrapper<T>(expected_value) << " op = " << ComparisonOpToString(op) << std::endl;
      num_failed++;
      if (exit_on_failure) return false;
    } else {
      if (syrah_debug_verbose) {
        std::cerr << VECMASK_STRING << " = " << ValWrapper<T>(a[i]) << " " << ComparisonOpToString(op) << " " << ValWrapper<T>(b[i]) << std::endl;
      }
    }
  }
  return num_failed == 0;
}

template <typename T, int N>
bool VecEqualsArrayTernaryOp(const FixedVector<T, N>& vec, const T* a, const T* b, const T* c, TernaryOp op) {
  int num_failed = 0;
  for (int i = 0; i < N; i++) {
    T expected_value = T(0);
    switch (op) {
    case MADD:
      expected_value = a[i] * b[i] + c[i];
      break;
    }
    if (!ValueEquals<T>(vec[i], expected_value)) {
      std::cerr << VEC_STRING << ". Expected " << ValWrapper<T>(expected_value) << "op = " << TernaryOpToString(op) << std::endl;
      num_failed++;
      if (exit_on_failure) return false;
    } else {
      if (syrah_debug_verbose) {
        std::cerr << VEC_STRING << " = " << TernaryOpToString(op) << " " << ValWrapper<T>(a[i]) << ", " << ValWrapper<T>(b[i]) << ", " << ValWrapper<T>(c[i]) << std::endl;
      }
    }
  }
  return num_failed == 0;
}

template <typename T, int N>
bool ScalarEqualReductionOp(const T& val, const T* a, ReductionOp op) {
  T expected_value = T(0);
  switch (op) {
  case ReduceMin:
    expected_value = a[0];
    break;
  case ReduceMax:
    expected_value = a[0];
    break;
  case ReduceSum:
    expected_value = T(0);
    break;
  case ReduceProduct:
    expected_value = T(1);
    break;
  }

  for (int i = 0; i < N; i++) {
    switch (op) {
    case ReduceMin:
      expected_value = std::min(expected_value, a[i]);
      break;
    case ReduceMax:
      expected_value = std::max(expected_value, a[i]);
      break;
    case ReduceSum:
      expected_value += a[i];
      break;
    case ReduceProduct:
      expected_value *= a[i];
      break;
    }
  }

  if (!ValueEquals<T>(val, expected_value)) {
    std::cerr << "Reduced val (" << TypeToString<T>() << "_" << LengthToString<N>() << ") = " << ValWrapper<T>(val) << ". Expected " << ValWrapper<T>(expected_value) << " op = " << ReductionOpToString(op) << std::endl;
    return false;
  } else {
    if (syrah_debug_verbose) {
      std::cerr << "ReducedVal = " << ValWrapper<T>(val) << " != " << ReductionOpToString(op) << " [";
      for (int i = 0; i < N; i++) {
        if (i != 0) std::cerr << ", ";
        std::cerr << ValWrapper<T>(a[i]);
      }
      std::cerr << "] = " << ValWrapper<T>(expected_value) << std::endl;
    }
  }
  return true;
}




template <typename T>
T ScalarFloor(const T val) {
  if (typeid(T) == typeid(float)) {
    return (T)floorf(val);
  } else if (typeid(T) == typeid(double)) {
    return (T)floor(val);
  } else if (typeid(T) == typeid(int) ||
             typeid(T) == typeid(uint8_t)) {
    return (T)(val);
  }
  return 0;
}

template <typename T>
T ScalarCeil(const T val) {
  if (typeid(T) == typeid(float)) {
    return (T)ceilf(val);
  } else if (typeid(T) == typeid(double)) {
    return (T)ceil(val);
  } else if (typeid(T) == typeid(int) ||
             typeid(T) == typeid(uint8_t)) {
    return (T)(val);
  }
  return 0;
}

template <typename T>
T ScalarFrac(const T val) {
  return val - ScalarFloor(val);
}


template <typename T, int N>
bool VecEqualsArrayUnaryOp(const FixedVector<T, N>& vec, const T* a, UnaryOp op) {
  int num_failed = 0;
  for (int i = 0; i < N; i++) {
    T expected_value = T(0);
    switch (op) {
    case NEG:
      expected_value = -a[i];
      break;
    case ABS:
      expected_value = std::abs(a[i]);
      break;
    case FLOOR:
      expected_value = ScalarFloor<T>(a[i]);
      break;
    case CEIL:
      expected_value = ScalarCeil<T>(a[i]);
      break;
    case FRAC:
      expected_value = ScalarFrac<T>(a[i]);
      break;
    case SIN:
      expected_value = std::sin(a[i]);
      break;
    case COS:
      expected_value = std::cos(a[i]);
      break;
    case TAN:
      expected_value = std::tan(a[i]);
      break;
    case COT:
      // lame, no actual function provided...
      expected_value = T(1)/std::tan(a[i]);
      break;
    case EXP:
      expected_value = std::exp(a[i]);
      break;
    case LN:
      expected_value = std::log(a[i]);
      break;
    case ATAN:
      expected_value = std::atan(a[i]);
      break;
    }

    if (!ValueEquals<T>(vec[i], expected_value)) {
      std::cerr << VEC_STRING << ". Expected " << ValWrapper<T>(expected_value) << " = " << UnaryOpToString(op) << " " << ValWrapper<T>(a[i]) << std::endl;
      num_failed++;
      if (exit_on_failure) return false;
    } else {
      if (syrah_debug_verbose) {
        std::cerr << VEC_STRING << " = " << UnaryOpToString(op) << " " << ValWrapper<T>(a[i]) << std::endl;
      }
    }
  }
  return num_failed == 0;
}


template <typename T, int N>
bool VecEqualsArrayPermuteOp(const FixedVector<T, N>& vec, const T* a, PermuteOp op) {
  int num_failed = 0;
  for (int i = 0; i < N; i++) {
    T expected_value = T(0);
    switch (op) {
    case Reverse:
      expected_value = a[N - 1 - i];
      break;
    }

    if (!ValueEquals<T>(vec[i], expected_value)) {
      std::cerr << VEC_STRING << ". Expected " << ValWrapper<T>(expected_value) << " = " << PermuteOpToString(op) << " " << ValWrapper<T>(a[i]) << std::endl;
      num_failed++;
      if (exit_on_failure) return false;
    } else {
      if (syrah_debug_verbose) {
        std::cerr << VEC_STRING << " = " << PermuteOpToString(op) << " " << ValWrapper<T>(a[i]) << std::endl;
      }
    }
  }
  return num_failed == 0;
}


template <typename T, int N>
bool TestConstantLoad(const T constant_value) {
  FixedVector<T, N> constant_vec(constant_value);
  return VecEqualsConstant<T, N>(constant_vec, constant_value);
}

template <typename T, int N>
bool TestArrayLoad(const T* values) {
  FixedVector<T, N> vec(values);
  return VecEqualsArray<T, N>(vec, values);
}

template <typename T, int N>
bool TestConstantAdd(const T a, const T b) {
  FixedVector<T, N> a_vec(a);
  FixedVector<T, N> b_vec(b);
  FixedVector<T, N> c_vec(a+b);
  T c_val(a+b);
  return VecEqualsConstant<T, N>(c_vec, c_val);
}

template <typename T, int N, typename OtherType>
bool TestConstantCast(const OtherType a) {
  FixedVector<T, N> a_vec(a);
  T expected_val(static_cast<T>(a));
  return VecEqualsConstant<T, N>(a_vec, expected_val);
}

template <int WIDTH>
bool ConstantTestPerWidth() {
  const float kFirstVal = 1.2345f;
  const float kSecondVal = 3.4567f;
  const double kFirstValDouble = static_cast<double>(kFirstVal);
  const double kSecondValDouble = static_cast<double>(kSecondVal);
  const int kFirstValInt = static_cast<int>(kFirstVal);
  const int kSecondValInt = static_cast<int>(kSecondVal);

  if (!TestConstantLoad<float, WIDTH>(kFirstVal)) return false;
  if (!TestConstantLoad<float, WIDTH>(kSecondVal)) return false;
  if (!TestConstantLoad<double, WIDTH>(kFirstValDouble)) return false;
  if (!TestConstantLoad<double, WIDTH>(kSecondValDouble)) return false;

  if (!TestConstantLoad<int, WIDTH>(kFirstValInt)) return false;
  if (!TestConstantLoad<int, WIDTH>(kSecondValInt)) return false;

  if (!TestConstantAdd<float, WIDTH>(kFirstVal, kSecondVal)) return false;
  if (!TestConstantAdd<double, WIDTH>(kFirstValDouble, kSecondValDouble)) return false;

  if (!TestConstantCast<float, WIDTH>(kFirstValDouble)) return false;
  if (!TestConstantCast<double, WIDTH>(kFirstVal)) return false;
  return true;
}

bool ConstantTests() {
  //if (!ConstantTestPerWidth<8>()) return false;
  if (!ConstantTestPerWidth<16>()) return false;
  if (!ConstantTestPerWidth<32>()) return false;
  return true;
}

const int kMaxWidth = 32;
SYRAH_ALIGN(16) float kFloatSequence[kMaxWidth] = {
  0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f,
  8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f,
  16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f,
  24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f,
};

SYRAH_ALIGN(16) double kDoubleSequence[kMaxWidth] = {
  0., 1., 2., 3., 4., 5., 6., 7.,
  8., 9., 10., 11., 12., 13., 14., 15.,
  16., 17., 18., 19., 20., 21., 22., 23.,
  24., 25., 26., 27., 28., 29., 30., 31.,
};

SYRAH_ALIGN(16) int kIntSequence[kMaxWidth] = {
  0, 1, 2, 3, 4, 5, 6, 7,
  8, 9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23,
  24, 25, 26, 27, 28, 29, 30, 31,
};

SYRAH_ALIGN(16) uint8_t kUInt8Sequence[kMaxWidth] = {
  0, 1, 2, 3, 4, 5, 6, 7,
  8, 9, 10, 11, 12, 13, 14, 15,
  16, 17, 18, 19, 20, 21, 22, 23,
  24, 25, 26, 27, 28, 29, 30, 31,
};


SYRAH_ALIGN(16) float kFloatOpA[kMaxWidth] = {
  1.f, 3.14159f, 2.71828f, 4.f, 22.f/7.f, 1.f/6.f, -1.f/3.f, 0.f,
  .693f, .707f, 1.732f, .3183f, 1e8f, 1.f/24.f, 1.f/48.f, 29.97f,
  255.f, 1.f/255.f, 2.2f, 1.f/2.2f, 1e-8f, 1.203f, 20.f, 2012.f,
  65536.f, -32767.f, 12345.f, 1048576.f, 0.5f, 0.25f, 0.125f, 0.0625f,
};

SYRAH_ALIGN(16) float kFloatOpB[kMaxWidth] = {
  2.f, 0.125f, 24.f, 1984.f, 9.f, 381.f, -100.f, 360.f,
  1.f, 5.f, 10.f, 25.f, 50.f, 100.f, 500.f, 1000.f,
  3.f, 5.f, 7.f, 11.f, 13.f, 17.f, 23.f, 29.f,
  -1.f/36.f, -5.f/36.f, 6500.f, 9300.f, 525.f, 800.f, 1024.f, 9.8f,
};

SYRAH_ALIGN(16) float kFloatOpC[kMaxWidth] = {
  -10.f, 32.f, .69234f, 19.84f, 100.f, 128.f, -200.f, 604.f,
   10.f, .23f, 2.812f, 19.76f,  .01f,  64.f,  -273.f, 603.f,
  -20.f,1.f/2.2f, 3.1459f, 19.28f, 1e-4f, 16.f, 343.f, 601.f,
  -40.f, .4545f, 6.28f, 20.00f, 1e-6f, 8.f, 500.f, 68e03f,
};

SYRAH_ALIGN(16) double kDoubleOpA[kMaxWidth] = {
  1., 3.14159, 2.71828, 4., 22./7., 1./6., -1./3., 0.,
  .693, .707, 1.732, .3183, 1e8, 1./24., 1./48., 29.97,
  255., 1./255., 2.2, 1./2.2, 1e-8, 1.203, 20., 2012.,
  65536., -32767., 12345., 1048576., 0.5, 0.25, 0.125, 0.0625,
};

SYRAH_ALIGN(16) double kDoubleOpB[kMaxWidth] = {
  2., 0.125, 24., 1984., 9., 381., -100., 360.,
  1., 5., 10., 25., 50., 100., 500., 1000.,
  3., 5., 7., 11., 13., 17., 23., 29.,
  -1./36., -5./36., 6500., 9300., 525., 800., 1024., 9.8,
};

SYRAH_ALIGN(16) double kDoubleOpC[kMaxWidth] = {
  -10., 32., .69234, 19.84, 100., 128., -200., 604.,
   10., .23, 2.812, 19.76,  .01,  64.,  -273., 603.,
  -20.,1./2.2, 3.1459, 19.28, 1e-4, 16., 343., 601.,
  -40., .4545, 6.28, 20.00, 1e-6, 8., 500., 68e03,
};


SYRAH_ALIGN(16) int kIntOpA[kMaxWidth] = {
  1, 3, 2, 4, 3, 6, -3, 1000,
  6, 7, 1, 3, 100000000, 24, 48, 30,
  255, 1, 2, 45, -10000, 1, 20, 2012,
  65536, -32767, 12345, 1048576, 2, 4, 8, 16,
};

SYRAH_ALIGN(16) int kIntOpB[kMaxWidth] = {
  2, 12, 24, 1984, 9, 381, -100, 360,
  1, 5, 10, 25, 50, 100, 500, 1000,
  3, 5, 7, 11, 13, 17, 23, 29,
  -36, -180, 6500, 9300, 525, 800, 1024, 10,
};

SYRAH_ALIGN(16) int kIntOpC[kMaxWidth] = {
  10, 32, 0, 19, 100, 128, -200, 604,
  10, 0, 3, 20, 0, 64, -273, 603,
  20, 45, 3, 19, 1000, 16, 343, 601,
  -40, 45, 6, 20, 100000, 8, 500, 68000
};


SYRAH_ALIGN(16) uint8_t kUInt8OpA[kMaxWidth] = {
  1, 3, 2, 4, 3, 6, 3, 100,
  6, 7, 1, 3, 100, 24, 48, 30,
  255, 1, 2, 45, 100, 1, 20, 201,
  65, 32, 123, 104, 2, 4, 8, 16,
};

SYRAH_ALIGN(16) uint8_t kUInt8OpB[kMaxWidth] = {
  2, 12, 24, 198, 9, 38, 10, 36,
  1, 5, 10, 25, 50, 100, 50, 100,
  3, 5, 7, 11, 13, 17, 23, 29,
  36, 180, 65, 93, 52, 80, 102, 10,
};

SYRAH_ALIGN(16) uint8_t kUInt8OpC[kMaxWidth] = {
  10, 32, 0, 19, 100, 128, 200, 60,
  10, 0, 3, 20, 0, 64, 27, 63,
  20, 45, 3, 19, 100, 16, 34, 61,
  40, 45, 6, 20, 100, 8, 50, 68
};


// NOTE(boulos): Want to use [0,4] in first 4, [0,8] in first 8, etc
// so that the gather indices can just be read in order for the
// test. Assume power of 2 for this though. (so [8, 16] and then [16,
// 32]).
SYRAH_ALIGN(16) int kGatherIndices[kMaxWidth] = {
  3, 1, 0, 2,
  5, 4, 7, 6,
  15, 8, 13, 9, 11, 10, 14, 12,
  30, 29, 16, 23, 27, 18, 20, 17, 25, 31, 24, 19, 21, 26, 28, 22
};

SYRAH_ALIGN(16) bool kGatherMask[kMaxWidth] = {
  true, false, false, true,
  false, false, false, true,
  true, true, false, false, false, false, true, false,
  true, false, true, false, true, true, true, true, false, false, true, true, false, false, false, true
};

SYRAH_ALIGN(16) int kScatterIndices[kMaxWidth] = {
  0, 2, 3, 1,
  6, 4, 7, 5,
  14, 10, 12, 11, 9, 8, 13, 15,
  31, 21, 17, 25, 27, 28, 30, 29, 26, 24, 19, 16, 23, 22, 18, 20,
};

SYRAH_ALIGN(16) bool kScatterMask[kMaxWidth] = {
  false, true, false, true,
  true, false, false, true,
  false, true, false, true, true, false, false, false,
  true, false, true, false, false, true, true, true, false, true, true, true, false, true, false, true
};

SYRAH_ALIGN(16) int kShiftIndicesInt[kMaxWidth] = {
  3, 1, 0, 2,
  5, 4, 7, 6,
  15, 8, 13, 9, 11, 10, 14, 12,
  30, 29, 16, 23, 27, 18, 20, 17, 25, 31, 24, 19, 21, 26, 28, 22
};

SYRAH_ALIGN(16) uint8_t kShiftIndicesUInt8[kMaxWidth] = {
  3, 1, 0, 2,
  5, 4, 7, 6,
  15, 8, 13, 9, 11, 10, 14, 12,
  30, 29, 16, 23, 27, 18, 20, 17, 25, 31, 24, 19, 21, 26, 28, 22
};

SYRAH_ALIGN(16) int kAll5sInt[kMaxWidth] = {
  5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5
};

SYRAH_ALIGN(16) uint8_t kAll5sUInt8[kMaxWidth] = {
  5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5,
  5, 5, 5, 5, 5, 5, 5, 5
};

template <typename T>
T* GetSequenceArray() {
  if (typeid(T) == typeid(float)) {
    return (T*)kFloatSequence;
  } else if (typeid(T) == typeid(double)) {
    return (T*)kDoubleSequence;
  } else if (typeid(T) == typeid(int)) {
    return (T*)kIntSequence;
  } else if (typeid(T) == typeid(uint8_t)) {
    return (T*)kUInt8Sequence;
  }
  return 0;
}

template <typename T>
T* GetOperatorInput(int which) {
  if (typeid(T) == typeid(float)) {
    switch (which) {
    case 0: return (T*)kFloatOpA;
    case 1: return (T*)kFloatOpB;
    default: return (T*)kFloatOpC;
    }
  } else if (typeid(T) == typeid(double)) {
    switch (which) {
    case 0: return (T*)kDoubleOpA;
    case 1: return (T*)kDoubleOpB;
    default: return (T*)kDoubleOpC;
    }
  } else if (typeid(T) == typeid(int)) {
    switch (which) {
    case 0: return (T*)kIntOpA;
    case 1: return (T*)kIntOpB;
    default: return (T*)kIntOpC;
    }
  } else if (typeid(T) == typeid(uint8_t)) {
    switch (which) {
    case 0: return (T*)kUInt8OpA;
    case 1: return (T*)kUInt8OpB;
    default: return (T*)kUInt8OpC;
    }
  }
  return 0;
}

template <typename T, int WIDTH>
bool LoadTestPerWidth() {
  // 3 aligned tests (two different sizes, one at start, one half way
  // in) and 1 unaligned (offset by 3 units)
  T* sequence = GetSequenceArray<T>();
  if (!TestArrayLoad<T,  WIDTH>(sequence)) return false;
  if (2*WIDTH < kMaxWidth) {
    if (!TestArrayLoad<T,  WIDTH>(sequence+WIDTH)) return false;
  }
  if (WIDTH + 3 < kMaxWidth) {
    if (!TestArrayLoad<T, WIDTH>(sequence+3)) return false;
  }

  return true;
}

bool LoadTests() {
  if (!LoadTestPerWidth<float, 8>()) return false;
  if (!LoadTestPerWidth<float, 16>()) return false;

  if (!LoadTestPerWidth<double, 8>()) return false;
  if (!LoadTestPerWidth<double, 16>()) return false;

  if (!LoadTestPerWidth<int, 8>()) return false;
  if (!LoadTestPerWidth<int, 16>()) return false;

  if (!LoadTestPerWidth<uint8_t, 16>()) return false;
  if (!LoadTestPerWidth<uint8_t, 32>()) return false;

  return true;
}

template <typename T, int WIDTH>
bool BinaryOpTestPerWidthTemplatedType() {
  T* kInputA = GetOperatorInput<T>(0);
  T* kInputB = GetOperatorInput<T>(1);
  T five_array[WIDTH];
  for (int i = 0; i < WIDTH; i++) { five_array[i] = T(5); }

  FixedVector<T, WIDTH> a(kInputA);
  FixedVector<T, WIDTH> b(kInputB);

  FixedVector<T, WIDTH> a_plus_b(a+b);
  FixedVector<T, WIDTH> a_plus_five(a+T(5));
  FixedVector<T, WIDTH> a_plus_five_reverse(T(5)+a);

  FixedVector<T, WIDTH> a_sub_b(a-b);
  FixedVector<T, WIDTH> a_sub_five(a-T(5));
  FixedVector<T, WIDTH> a_sub_five_reverse(T(5) - a);

  FixedVector<T, WIDTH> a_times_b(a*b);
  FixedVector<T, WIDTH> a_times_5(a*T(5));
  FixedVector<T, WIDTH> a_times_5_reverse(T(5)*a);

  FixedVector<T, WIDTH> a_div_b(a/b);
  FixedVector<T, WIDTH> a_div_five(a/T(5));
  FixedVector<T, WIDTH> a_div_five_reverse(T(5)/a);

  FixedVector<T, WIDTH> min_a_b(min(a,b));
  FixedVector<T, WIDTH> max_a_b(max(a,b));

  if (!VecEqualsArrayBinaryOp<T, WIDTH>(a_plus_b, kInputA, kInputB, ADD)) return false;
  if (!VecEqualsArrayBinaryOp<T, WIDTH>(a_plus_five, kInputA, five_array, ADD)) return false;
  if (!VecEqualsArrayBinaryOp<T, WIDTH>(a_plus_five_reverse, five_array, kInputA, ADD)) return false;

  if (!VecEqualsArrayBinaryOp<T, WIDTH>(a_sub_b, kInputA, kInputB, SUB)) return false;
  if (!VecEqualsArrayBinaryOp<T, WIDTH>(a_sub_five, kInputA, five_array, SUB)) return false;
  if (!VecEqualsArrayBinaryOp<T, WIDTH>(a_sub_five_reverse, five_array, kInputA, SUB)) return false;

  if (!VecEqualsArrayBinaryOp<T, WIDTH>(a_times_b, kInputA, kInputB, MUL)) return false;
  if (!VecEqualsArrayBinaryOp<T, WIDTH>(a_times_5, kInputA, five_array, MUL)) return false;
  if (!VecEqualsArrayBinaryOp<T, WIDTH>(a_times_5_reverse, five_array, kInputA, MUL)) return false;

  if (!VecEqualsArrayBinaryOp<T, WIDTH>(a_div_b, kInputA, kInputB, DIV)) return false;
  if (!VecEqualsArrayBinaryOp<T, WIDTH>(a_div_five, kInputA, five_array, DIV)) return false;
  if (!VecEqualsArrayBinaryOp<T, WIDTH>(a_div_five_reverse, five_array, kInputA, DIV)) return false;

  if (!VecEqualsArrayBinaryOp<T, WIDTH>(min_a_b, kInputA, kInputB, MIN)) return false;
  if (!VecEqualsArrayBinaryOp<T, WIDTH>(max_a_b, kInputA, kInputB, MAX)) return false;

  return true;
}

template <int WIDTH>
bool BinaryOpTestPerWidth() {
  if (!BinaryOpTestPerWidthTemplatedType<float, WIDTH>()) return false;
  if (!BinaryOpTestPerWidthTemplatedType<double, WIDTH>()) return false;
  if (!BinaryOpTestPerWidthTemplatedType<int, WIDTH>()) return false;
  if (!BinaryOpTestPerWidthTemplatedType<uint8_t, WIDTH>()) return false;

  return true;
}

bool BinaryOpTests() {
  //if (!BinaryOpTestPerWidth<4>()) return false;
  //if (!BinaryOpTestPerWidth<8>()) return false;
  if (!BinaryOpTestPerWidth<16>()) return false;
  if (!BinaryOpTestPerWidth<32>()) return false;

  return true;
}

template <typename T, int WIDTH>
bool BinaryFloatOpTestPerWidthTemplatedType() {
  T* kInputA = GetOperatorInput<T>(0);
  T* kInputB = GetOperatorInput<T>(1);

  FixedVector<T, WIDTH> a(kInputA);
  FixedVector<T, WIDTH> b(kInputB);

  FixedVector<T, WIDTH> a_pow_b = pow(a, b);
  FixedVector<T, WIDTH> a_atan2_b = atan2(a, b);

  if (!VecEqualsArrayBinaryFloatOp<T, WIDTH>(a_pow_b, kInputA, kInputB, POWF)) return false;
  if (!VecEqualsArrayBinaryFloatOp<T, WIDTH>(a_atan2_b, kInputA, kInputB, ATAN2)) return false;
  return true;
}

template <int WIDTH>
bool BinaryFloatOpTestPerWidth() {
  if (!BinaryFloatOpTestPerWidthTemplatedType<float, WIDTH>()) return false;

  // Can't implement the math ops without 64-bit int support...

  //if (!BinaryFloatOpTestPerWidthTemplatedType<double, WIDTH>()) return false;

  return true;
}

bool BinaryFloatOpTests() {
  //if (!BinaryFloatOpTestPerWidth<8>()) return false;
  if (!BinaryFloatOpTestPerWidth<16>()) return false;
  if (!BinaryFloatOpTestPerWidth<32>()) return false;

  return true;
}


template <typename T, int WIDTH>
bool UnaryOpTestPerWidthTemplatedType() {
  T* kInputA = GetOperatorInput<T>(0);

  FixedVector<T, WIDTH> a(kInputA);

  FixedVector<T, WIDTH> neg_a(-a);
  FixedVector<T, WIDTH> abs_a(abs(a));
  FixedVector<T, WIDTH> floor_a(floor(a));
  FixedVector<T, WIDTH> ceil_a(ceil(a));
  FixedVector<T, WIDTH> frac_a(frac(a));

  FixedVector<T, WIDTH> sin_a(sin(a));
  FixedVector<T, WIDTH> cos_a(cos(a));
  FixedVector<T, WIDTH> tan_a(tan(a));
  FixedVector<T, WIDTH> sincos_sin_a;
  FixedVector<T, WIDTH> sincos_cos_a;
  sincos(a, sincos_sin_a, sincos_cos_a);

  FixedVector<T, WIDTH> exp_a(exp(a));
  FixedVector<T, WIDTH> ln_a(ln(a));
  FixedVector<T, WIDTH> atan_a(atan(a));

  int num_failed = 0;

  if (!VecEqualsArrayUnaryOp<T, WIDTH>(neg_a,   kInputA, NEG)) { num_failed++; if (exit_on_failure) return false; }
  if (!VecEqualsArrayUnaryOp<T, WIDTH>(abs_a,   kInputA, ABS)) { num_failed++; if (exit_on_failure) return false; }
  if (!VecEqualsArrayUnaryOp<T, WIDTH>(floor_a, kInputA, FLOOR)) { num_failed++; if (exit_on_failure) return false; }
  if (!VecEqualsArrayUnaryOp<T, WIDTH>(ceil_a,  kInputA, CEIL)) { num_failed++; if (exit_on_failure) return false; }
  if (!VecEqualsArrayUnaryOp<T, WIDTH>(frac_a,  kInputA, FRAC)) { num_failed++; if (exit_on_failure) return false; }

  bool is_float_type = (typeid(T) == typeid(float) || typeid(T) == typeid(double));

  if (is_float_type && !VecEqualsArrayUnaryOp<T, WIDTH>(sin_a,  kInputA, SIN)) { num_failed++; if (exit_on_failure) return false; }
  if (is_float_type && !VecEqualsArrayUnaryOp<T, WIDTH>(cos_a,  kInputA, COS)) { num_failed++; if (exit_on_failure) return false; }
  if (is_float_type && !VecEqualsArrayUnaryOp<T, WIDTH>(tan_a,  kInputA, TAN)) { num_failed++; if (exit_on_failure) return false; }
  if (is_float_type && !VecEqualsArrayUnaryOp<T, WIDTH>(sincos_sin_a, kInputA, SIN)) { num_failed++; if (exit_on_failure) return false; }
  if (is_float_type && !VecEqualsArrayUnaryOp<T, WIDTH>(sincos_cos_a, kInputA, COS)) { num_failed++; if (exit_on_failure) return false; }
  if (is_float_type && !VecEqualsArrayUnaryOp<T, WIDTH>(exp_a, kInputA, EXP)) { num_failed++; if (exit_on_failure) return false; }
  if (is_float_type && !VecEqualsArrayUnaryOp<T, WIDTH>(ln_a, kInputA, LN)) { num_failed++; if (exit_on_failure) return false; }
  if (is_float_type && !VecEqualsArrayUnaryOp<T, WIDTH>(atan_a, kInputA, ATAN)) { num_failed++; if (exit_on_failure) return false; }

  return num_failed == 0;
}

template <int WIDTH>
bool UnaryOpTestPerWidth() {
  int num_failed = 0;
  if (!UnaryOpTestPerWidthTemplatedType<float, WIDTH>()) num_failed++;

  // TODO(boulos): Need 64-bit int to implement these properly! (need
  //reinterpret and such)

  //if (!UnaryOpTestPerWidthTemplatedType<double, WIDTH>()) return false;

  // Test unary negate for int separately
  int* kInputA = GetOperatorInput<int>(0);
  FixedVector<int, WIDTH> a(kInputA);
  FixedVector<int, WIDTH> neg_a(-a);
  FixedVector<int, WIDTH> abs_a(abs(a));

  if (!VecEqualsArrayUnaryOp<int, WIDTH>(neg_a, kInputA, NEG)) num_failed++;
  if (!VecEqualsArrayUnaryOp<int, WIDTH>(abs_a, kInputA, ABS)) num_failed++;

  return num_failed == 0;
}

bool UnaryOpTests() {
  //if (!UnaryOpTestPerWidth<4>()) return false;
  //if (!UnaryOpTestPerWidth<8>()) return false;
  if (!UnaryOpTestPerWidth<16>()) return false;
  if (!UnaryOpTestPerWidth<32>()) return false;

  return true;
}

template <typename T, int WIDTH>
bool TernaryOpTestPerWidthTemplatedType() {
  T* kInputA = GetOperatorInput<T>(0);
  T* kInputB = GetOperatorInput<T>(1);
  T* kInputC = GetOperatorInput<T>(2);

  FixedVector<T, WIDTH> a(kInputA);
  FixedVector<T, WIDTH> b(kInputB);
  FixedVector<T, WIDTH> c(kInputC);

  FixedVector<T, WIDTH> madd_value(madd(a,b,c));
  FixedVector<T, WIDTH> madd_equiv(a*b + c);

  if (!VecEqualsArrayTernaryOp<T, WIDTH>(madd_value, kInputA, kInputB, kInputC, MADD)) return false;
  if (!VecEqualsArrayTernaryOp<T, WIDTH>(madd_equiv, kInputA, kInputB, kInputC, MADD)) return false;

  return true;
}

template <int WIDTH>
bool TernaryOpTestPerWidth() {
  int num_failed = 0;
  if (!TernaryOpTestPerWidthTemplatedType<float, WIDTH>()) { num_failed++; if (exit_on_failure) return false; }
  if (!TernaryOpTestPerWidthTemplatedType<double, WIDTH>()) { num_failed++; if (exit_on_failure) return false; }
  if (!TernaryOpTestPerWidthTemplatedType<int, WIDTH>()) {num_failed++; if (exit_on_failure) return false; }

  return num_failed == 0;
}

bool TernaryOpTests() {
  //if (!TernaryOpTestPerWidth<4>()) return false;
  //if (!TernaryOpTestPerWidth<8>()) return false;
  if (!TernaryOpTestPerWidth<16>()) return false;
  if (!TernaryOpTestPerWidth<32>()) return false;

  return true;
}


// Comparisons
template <typename T, int WIDTH>
bool ComparisonOpTestPerWidthTemplatedType() {
  T* kInputA = GetOperatorInput<T>(0);
  T* kInputB = GetOperatorInput<T>(1);

  FixedVector<T, WIDTH> a(kInputA);
  FixedVector<T, WIDTH> b(kInputB);

  FixedVectorMask<WIDTH> a_lt_b(a < b);
  FixedVectorMask<WIDTH> a_le_b(a <= b);
  FixedVectorMask<WIDTH> a_eq_b(a == b);
  FixedVectorMask<WIDTH> a_ge_b(a >= b);
  FixedVectorMask<WIDTH> a_gt_b(a > b);

  FixedVectorMask<WIDTH> a_le_b_equiv_or(a_lt_b | a_eq_b);
  FixedVectorMask<WIDTH> a_le_b_equiv_cmp(!a_gt_b);

  if (!VecEqualsArrayComparisonOp<T, WIDTH>(a_lt_b, kInputA, kInputB, LT)) return false;
  if (!VecEqualsArrayComparisonOp<T, WIDTH>(a_le_b, kInputA, kInputB, LE)) return false;
  if (!VecEqualsArrayComparisonOp<T, WIDTH>(a_eq_b, kInputA, kInputB, EQ)) return false;
  if (!VecEqualsArrayComparisonOp<T, WIDTH>(a_ge_b, kInputA, kInputB, GE)) return false;
  if (!VecEqualsArrayComparisonOp<T, WIDTH>(a_gt_b, kInputA, kInputB, GT)) return false;

  if (!VecEqualsArrayComparisonOp<T, WIDTH>(a_le_b_equiv_or, kInputA, kInputB, LE)) return false;
  if (!VecEqualsArrayComparisonOp<T, WIDTH>(a_le_b_equiv_cmp, kInputA, kInputB, LE)) return false;

  return true;
}

template <int WIDTH>
bool ComparisonOpTestPerWidth() {
  if (!ComparisonOpTestPerWidthTemplatedType<float, WIDTH>()) return false;
  if (!ComparisonOpTestPerWidthTemplatedType<double, WIDTH>()) return false;
  if (!ComparisonOpTestPerWidthTemplatedType<int, WIDTH>()) return false;

  return true;
}

bool ComparisonOpTests() {
  //if (!ComparisonOpTestPerWidth<4>()) return false;
  //if (!ComparisonOpTestPerWidth<8>()) return false;
  if (!ComparisonOpTestPerWidth<16>()) return false;
  if (!ComparisonOpTestPerWidth<32>()) return false;

  return true;
}

template <typename T, int WIDTH>
bool MergeSelectTestPerWidthTemplatedType() {
  T* kInputA = GetOperatorInput<T>(0);
  T* kInputB = GetOperatorInput<T>(1);

  FixedVector<T, WIDTH> a(kInputA);
  FixedVector<T, WIDTH> b(kInputB);

  FixedVectorMask<WIDTH> a_le_b(a <= b);
  FixedVectorMask<WIDTH> a_ge_b(a >= b);

  FixedVector<T, WIDTH> min_a_b(min(a,b));
  FixedVector<T, WIDTH> min_equiv(select(a, b, a_le_b));
  FixedVector<T, WIDTH> min_equiv_merge(b);
  min_equiv_merge.merge(a, a_le_b);

  FixedVector<T, WIDTH> max_a_b(min(a,b));
  FixedVector<T, WIDTH> max_equiv(select(a, b, a_ge_b));
  FixedVector<T, WIDTH> max_equiv_merge(b);
  max_equiv_merge.merge(a, a_ge_b);

  if (!VecEqualsArrayBinaryOp<T, WIDTH>(min_equiv, kInputA, kInputB, MIN)) return false;
  if (!VecEqualsArrayBinaryOp<T, WIDTH>(min_equiv_merge, kInputA, kInputB, MIN)) return false;

  if (!VecEqualsArrayBinaryOp<T, WIDTH>(max_equiv, kInputA, kInputB, MAX)) return false;
  if (!VecEqualsArrayBinaryOp<T, WIDTH>(max_equiv_merge, kInputA, kInputB, MAX)) return false;

  return true;
}

template <int WIDTH>
bool MergeSelectTestPerWidth() {
  if (!MergeSelectTestPerWidthTemplatedType<float, WIDTH>()) return false;

#ifndef __AVX__ // generates an ICE with latest ICC. Thanks Intel.
  if (!MergeSelectTestPerWidthTemplatedType<double, WIDTH>()) return false;
#endif
  if (!MergeSelectTestPerWidthTemplatedType<int, WIDTH>()) return false;

  return true;
}

bool MergeSelectTests() {
  //if (!MergeSelectTestPerWidth<4>()) return false;
  //if (!MergeSelectTestPerWidth<8>()) return false;
  if (!MergeSelectTestPerWidth<16>()) return false;
  if (!MergeSelectTestPerWidth<32>()) return false;

  return true;
}

template <typename T, int N>
bool GatherScatterTestPerWidthTemplatedType() {
  T* kInputA = GetOperatorInput<T>(0);
  int num_failed = 0;
  {
    FixedVector<int, N> gather_indices(kGatherIndices);
    FixedVector<T, N> vec(kInputA, gather_indices, sizeof(T));
    for (int i = 0; i < N; i++) {
      T expected = kInputA[gather_indices[i]];
      if (!ValueEquals<T>(vec[i], expected)) {
        std::cerr << VEC_STRING << ". Expected " << ValWrapper<T>(expected) << " op = gather " << std::endl;
        num_failed++;
        if (exit_on_failure) return false;
      }
    }
  }

  {
    FixedVectorMask<N> gather_mask(kGatherMask);
    FixedVector<int, N> gather_indices(kGatherIndices);
    FixedVector<T, N> vec = FixedVector<T, N>::Zero();
    vec.gather(kInputA, gather_indices, static_cast<int>(sizeof(T)), gather_mask);
    for (int i = 0; i < N; i++) {
      T expected = (kGatherMask[i]) ? kInputA[gather_indices[i]] : 0;
      if (!ValueEquals<T>(vec[i], expected)) {
        std::cerr << VEC_STRING << ". Expected << " << ValWrapper<T>(expected) << " op = masked_gather" << std::endl;
        num_failed++;
        if (exit_on_failure) return false;
      }
    }
  }

  {
    FixedVector<int, N> scatter_indices(kScatterIndices);
    FixedVector<T, N> scatter_data(kInputA);
    SYRAH_ALIGN(16) T scatter_dest[kMaxWidth];
    for (int i = 0; i < N; i++) {
      scatter_dest[i] = static_cast<T>(0);
    }

    scatter_data.scatter(scatter_dest, scatter_indices, static_cast<int>(sizeof(T)));
    for (int i = 0; i < N; i++) {
      T result = scatter_dest[scatter_indices[i]];
      T expected = kInputA[i];
      if (result != expected) {
        std::cerr << "scattered_" << LengthToString<N>() << "_" << TypeToString<T>() << "[" << i << "] = " << ValWrapper<T>(result) << " but expected " << ValWrapper<T>(expected) << std::endl;
        num_failed++;
        if (exit_on_failure) return false;
      }
    }
  }

  {
    FixedVector<int, N> scatter_indices(kScatterIndices);
    FixedVectorMask<N> scatter_mask(kScatterMask);
    FixedVector<T, N> scatter_data(kInputA);
    SYRAH_ALIGN(16) T scatter_dest[kMaxWidth];
    for (int i = 0; i < N; i++) {
      scatter_dest[i] = static_cast<T>(0);
    }
    scatter_data.scatter(scatter_dest, scatter_indices, sizeof(T), scatter_mask);
    for (int i = 0; i < N; i++) {
      T result = scatter_dest[scatter_indices[i]];
      T expected = (kScatterMask[i]) ? kInputA[i] : 0;
      if (result != expected) {
        std::cerr << "masked_scatter_" << LengthToString<N>() << "_" << TypeToString<T>() << "[" << i << "] = " << ValWrapper<T>(result) << " but expected " << ValWrapper<T>(expected) << std::endl;
        num_failed++;
        if (exit_on_failure) return false;
      }
    }
  }

  return num_failed == 0;
}

template <int WIDTH>
bool GatherScatterTestPerWidth() {
  if (!GatherScatterTestPerWidthTemplatedType<float, WIDTH>()) return false;
  if (!GatherScatterTestPerWidthTemplatedType<double, WIDTH>()) return false;
  if (!GatherScatterTestPerWidthTemplatedType<int, WIDTH>()) return false;
  if (!GatherScatterTestPerWidthTemplatedType<uint8_t, WIDTH>()) return false;

  return true;
}

bool GatherScatterTests() {
  //if (!GatherScatterTestPerWidth<4>()) return false;
  //if (!GatherScatterTestPerWidth<8>()) return false;
  if (!GatherScatterTestPerWidth<16>()) return false;
  if (!GatherScatterTestPerWidth<32>()) return false;

  return true;
}

// Reduction ops
template <typename T, int WIDTH>
bool ReductionTestPerWidthTemplatedType() {
  T* kInputA = GetOperatorInput<T>(0);

  FixedVector<T, WIDTH> a(kInputA);

  T hmax_a = a.foldMax();
  T hmin_a = a.foldMin();
  T hsum_a = a.foldSum();
  T hmul_a = a.foldProd();

  int num_failed = 0;

  if (!ScalarEqualReductionOp<T, WIDTH>(hmax_a, kInputA, ReduceMax)) { num_failed++; }
  if (!ScalarEqualReductionOp<T, WIDTH>(hmin_a, kInputA, ReduceMin)) { num_failed++; }
  if (!ScalarEqualReductionOp<T, WIDTH>(hsum_a, kInputA, ReduceSum)) { num_failed++; }
  if (!ScalarEqualReductionOp<T, WIDTH>(hmul_a, kInputA, ReduceProduct)) { num_failed++; }

  return num_failed == 0;
}

template <int WIDTH>
bool ReductionTestPerWidth() {
  int num_failed = 0;
  if (!ReductionTestPerWidthTemplatedType<float, WIDTH>()) num_failed++;
  if (!ReductionTestPerWidthTemplatedType<double, WIDTH>()) num_failed++;
  if (!ReductionTestPerWidthTemplatedType<int, WIDTH>()) num_failed++;
  if (!ReductionTestPerWidthTemplatedType<uint8_t, WIDTH>()) num_failed++;

  return num_failed == 0;
}

bool ReductionTests() {
  int num_failed = 0;
   //if (!ReductionTestPerWidth<8>()) return false;
  if (!ReductionTestPerWidth<16>()) num_failed++;
  if (!ReductionTestPerWidth<32>()) num_failed++;
  return num_failed == 0;
}

template<typename T, int WIDTH>
bool BitwiseTestPerWidthTemplatedType() {
  T* kInputA = GetOperatorInput<T>(0);
  T* kInputB = GetOperatorInput<T>(1);

  T* kShiftIndices = (typeid(T) == typeid(int)) ? (T*)kShiftIndicesInt : (T*)kShiftIndicesUInt8;
  T* kAll5s = (typeid(T) == typeid(int)) ? (T*)kAll5sInt : (T*)kAll5sUInt8;

  FixedVector<T, WIDTH> a(kInputA);
  FixedVector<T, WIDTH> b(kInputB);
  FixedVector<T, WIDTH> shift_indices(kShiftIndices);

  // AND(=), OR(=), XOR(=), NOT, COMP?
  FixedVector<T, WIDTH> a_and_b = a & b;
  FixedVector<T, WIDTH> a_or_b = a | b;
  FixedVector<T, WIDTH> a_xor_b = a ^ b;
  FixedVector<T, WIDTH> a_shl_b = a << shift_indices;
  FixedVector<T, WIDTH> a_shr_b = a >> shift_indices;

  FixedVector<T, WIDTH> a_shl_5 = a << 5;
  FixedVector<T, WIDTH> a_shr_5 = a >> 5;

  FixedVector<T, WIDTH> a_andeq_b = a; a_andeq_b &= b;
  FixedVector<T, WIDTH> a_oreq_b = a; a_oreq_b |= b;
  FixedVector<T, WIDTH> a_xoreq_b = a; a_xoreq_b ^= b;
  FixedVector<T, WIDTH> a_shleq_b = a; a_shleq_b <<= shift_indices;
  FixedVector<T, WIDTH> a_shreq_b = a; a_shreq_b >>= shift_indices;

  FixedVector<T, WIDTH> a_shleq_5 = a; a_shleq_5 <<= 5;
  FixedVector<T, WIDTH> a_shreq_5 = a; a_shreq_5 >>= 5;

  if (!VecEqualsArrayBitwiseOp<T, WIDTH>(a_and_b, kInputA, kInputB, AND)) return false;
  if (!VecEqualsArrayBitwiseOp<T, WIDTH>(a_or_b, kInputA, kInputB, OR)) return false;
  if (!VecEqualsArrayBitwiseOp<T, WIDTH>(a_xor_b, kInputA, kInputB, XOR)) return false;
  if (!VecEqualsArrayBitwiseOp<T, WIDTH>(a_shl_b, kInputA, kShiftIndices, SHL)) return false;
  if (!VecEqualsArrayBitwiseOp<T, WIDTH>(a_shr_b, kInputA, kShiftIndices, SHR)) return false;

  if (!VecEqualsArrayBitwiseOp<T, WIDTH>(a_shl_5, kInputA, kAll5s, SHL)) return false;
  if (!VecEqualsArrayBitwiseOp<T, WIDTH>(a_shr_5, kInputA, kAll5s, SHR)) return false;

  if (!VecEqualsArrayBitwiseOp<T, WIDTH>(a_andeq_b, kInputA, kInputB, ANDEQ)) return false;
  if (!VecEqualsArrayBitwiseOp<T, WIDTH>(a_oreq_b, kInputA, kInputB, OREQ)) return false;
  if (!VecEqualsArrayBitwiseOp<T, WIDTH>(a_xoreq_b, kInputA, kInputB, XOREQ)) return false;
  if (!VecEqualsArrayBitwiseOp<T, WIDTH>(a_shleq_b, kInputA, kShiftIndices, SHLEQ)) return false;
  if (!VecEqualsArrayBitwiseOp<T, WIDTH>(a_shreq_b, kInputA, kShiftIndices, SHREQ)) return false;

  if (!VecEqualsArrayBitwiseOp<T, WIDTH>(a_shleq_5, kInputA, kAll5s, SHLEQ)) return false;
  if (!VecEqualsArrayBitwiseOp<T, WIDTH>(a_shreq_5, kInputA, kAll5s, SHREQ)) return false;

  return true;
}

template<int WIDTH>
bool BitwiseTestPerWidth() {
  // TODO(boulos): Add unsigned int, long, unsigned long long, short,
  // char, etc (and reinterpret cast!)
  if (!BitwiseTestPerWidthTemplatedType<int, WIDTH>()) return false;
  if (!BitwiseTestPerWidthTemplatedType<uint8_t, WIDTH>()) return false;
  return true;
}

bool BitwiseTests() {
   //if (!BitwiseTestPerWidth<8>()) return false;
  if (!BitwiseTestPerWidth<16>()) return false;
  if (!BitwiseTestPerWidth<32>()) return false;
  return true;
}

template<typename T, int WIDTH>
bool PermuteTestsPerWidthTemplatedType() {
  T* kInputA = GetOperatorInput<T>(0);

  // Test reverse
  FixedVector<T, WIDTH> reversed = reverse(FixedVector<T, WIDTH>(kInputA));
  if (!VecEqualsArrayPermuteOp(reversed, kInputA, Reverse)) return false;

  return true;
}

template<int WIDTH>
bool PermuteTestsPerWidth() {
  if (!PermuteTestsPerWidthTemplatedType<float, WIDTH>()) return false;
  if (!PermuteTestsPerWidthTemplatedType<double, WIDTH>()) return false;
  if (!PermuteTestsPerWidthTemplatedType<int, WIDTH>()) return false;

  return true;
}

bool PermuteTests() {
  if (!PermuteTestsPerWidth<16>()) return false;
  if (!PermuteTestsPerWidth<32>()) return false;
  return true;
}

// TODO(boulos): PrefixSum

// TODO(boulos): Int shifts

// TODO(boulos): Conditional Load/Store (not just maksed gather/scatter)

// TODO(boulos): Mask ops (Any/All/None/And/Or/Xor/FirstN/Not)

int vec_tests() {
  syrah::DisableDenormals();
  std::cerr << std::setprecision(9);
  int num_failed = 0;
  if (!ConstantTests()) {
    std::cerr << "Constant Tests Failed." << std::endl;
    num_failed++;
    if (exit_on_failure) return -1;
  }

  if (!LoadTests()) {
    std::cerr << "Load tests failed." << std::endl;
    num_failed++;
    if (exit_on_failure) return -1;
  }

  if (!BinaryOpTests()) {
    std::cerr << "Binary op tests failed." << std::endl;
    num_failed++;
    if (exit_on_failure) return -1;
  }

  if (!BinaryFloatOpTests()) {
    std::cerr << "Binary float op tests failed." << std::endl;
    num_failed++;
    if (exit_on_failure) return -1;
  }

  if (!BitwiseTests()) {
    std::cerr << "Bitwise op tests failed." << std::endl;
    num_failed++;
    if (exit_on_failure) return -1;
  }

  if (!UnaryOpTests()) {
    std::cerr << "Unary op tests failed." << std::endl;
    num_failed++;
    if (exit_on_failure) return -1;
  }

  if (!TernaryOpTests()) {
    std::cerr << "Ternary op tests failed." << std::endl;
    num_failed++;
    if (exit_on_failure) return -1;
  }

  if (!ComparisonOpTests()) {
    std::cerr << "Comparison op tests failed." << std::endl;
    num_failed++;
    if (exit_on_failure) return -1;
  }

  if (!MergeSelectTests()) {
    std::cerr << "Merge/Select tests failed." << std::endl;
    num_failed++;
    if (exit_on_failure) return -1;
  }

  if (!GatherScatterTests()) {
    std::cerr << "Gather/Scatter tests failed." << std::endl;
    num_failed++;
    if (exit_on_failure) return -1;
  }

  if (!ReductionTests()) {
    std::cerr << "Reduction tests failed." << std::endl;
    num_failed++;
    if (exit_on_failure) return -1;
  }

  if (!PermuteTests()) {
    std::cerr << "Permute tests failed." << std::endl;
    num_failed++;
    if (exit_on_failure) return -1;
  }

  if (num_failed) {
    std::cerr << num_failed << " tests failed." << std::endl;
    return -1;
  }

  std::cerr << "All tests passed." << std::endl;
  return 0;
}

#if !(defined(__APPLE__) && defined(__ARM_NEON__)) && !(defined(__LRB__))
int main() {
  return vec_tests();
}
#endif

#if 0

int main() {
  FixedVectorMask<8> is_multiple_of_two(floor_result == sequence_div_two);
  FixedVector<float, 8> sequence_squared(eight_sequence * eight_sequence);
  sequence_squared.store(eight_sequence_b, is_multiple_of_two);
  printf("Conditional Store\n");
  for (int i = 0; i < 8; i++) {
    printf("%d: seq = %10.4f, seq^2 = %10.4f, seq^2.store(dest, seq.isMultipleOfTwo) = %10.f\n",
           i, eight_sequence[i], sequence_squared[i], eight_sequence_b[i]);
  }

  printf("NumActive, PrefixSum\n");
  int num_active = NumActive(gather_scatter_mask);
  FixedVector<int, 8> prefix_sum(PrefixSum(gather_scatter_mask));
  printf("Found %d Active\n", num_active);
  for (int i = 0; i < 8; i++) {
    printf("%d: mask = %s, prefix_sum = %d\n",
           i,
           gather_scatter_mask.get(i) ? "T" : "F",
           prefix_sum[i]);
  }


  int_tests();

  return 0;
}

void int_tests() {
  SYRAH_ALIGN(16) int shift_a_indices[8] = {
    1, 1, 2, 2, 4, 4, 8, 8
  };

  SYRAH_ALIGN(16) int shift_b_indices[8] = {
    0, 1, 2, 3, 4, 5, 6, 7
  };

  FixedVector<int, 8> a(shift_a_indices);
  FixedVector<int, 8> b(shift_b_indices);
  FixedVector<int, 8> a_sll_b = a << b;
  FixedVector<int, 8> b_sll_a = b << a;

  printf("Shift Test\n");
  for (int i = 0; i < 8; i++) {
    printf("%d: a = %5d, b= %5d, a << b = %5d, b << a = %5d\n",
           i, a[i], b[i], a_sll_b[i], b_sll_a[i]);
  }

  FixedVectorMask<8> a_eq_b(a == b);
  FixedVectorMask<8> a_neq_b(a != b);
  printf("Comparisons\n");
  for (int i = 0; i < 8; i++) {
    printf("%d: a = %d, b= %d, a == b = %s, a != b = %s\n",
           i, a[i], b[i],
                   a_eq_b.get(i)  ? "T" : "F",
                   a_neq_b.get(i) ? "T" : "F");
  }
}
#endif
