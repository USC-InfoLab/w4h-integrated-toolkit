#include <math.h>

#include "postgres.h"

#include "fmgr.h"
#include "libpq/pqformat.h"

#include "catalog/pg_type.h"
#include "common/int.h"
#include "common/pg_prng.h"
#include "common/shortest_dec.h"
#include "libpq/pqformat.h"
#include "miscadmin.h"
#include "utils/array.h"
#include "utils/float.h"
#include "utils/fmgrprotos.h"
#include "utils/sortsupport.h"
#include "utils/timestamp.h"

#include "avg.h"

#define TOPK 100

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(complex_avg);
PG_FUNCTION_INFO_V1(complex_avg_accum);
  
// source: https://github.com/postgres/postgres/blob/master/src/backend/utils/adt/float.c
static float8 *check_float8_array(ArrayType *transarray, const char *caller, int n) {
  /*
  * We expect the input to be an N-element float array; verify that. We
  * don't need to use deconstruct_array() since the array data is just
  * going to look like a C array of N float8 values.
  */
  if (ARR_NDIM(transarray) != 1 ||
    ARR_DIMS(transarray)[0] != n ||
    ARR_HASNULL(transarray) ||
    ARR_ELEMTYPE(transarray) != FLOAT8OID)
    elog(ERROR, "%s: expected %d-element float8 array", caller, n);
  return (float8 *) ARR_DATA_PTR(transarray);
}

/// AVG on FFT transformation
static float8 avg_integral_real(float8 N, float8 L, Complex* val, int32 t1, int32 t2) {
  float8 res;
  float8 mid = L/2;
  float8 period = t2-t1;
  if (N == 0.0) {
    res = 1/L * val->x * period;
  } else if (N == TOPK+1){
    float8 a = 1/L;
    float8 b = (2*M_PI*mid)/L;
    float8 y2 = val->x * sin(b*t2) + val->y * cos(b*t2);
    float8 y1 = val->x * sin(b*t1) + val->y * cos(b*t1);
    res = a * (1/b) * (y2-y1);
  } else {
    float8 a = 2/L;
    float8 b = (2*M_PI*N)/L;
    float8 y2 = val->x * sin(b*t2) + val->y * cos(b*t2);
    float8 y1 = val->x * sin(b*t1) + val->y * cos(b*t1);
    res = a * (1/b) * (y2-y1);
  }
  return res;
}

static float8 avg_integral_imag(float8 N, float8 L, Complex* val, int32 t1, int32 t2) {
  float8 res;
  float8 mid = L/2;
  float8 period = t2-t1;
  if (N == 0.0) {
    res = 1/L * val->y * period;
  } else if (N == TOPK+1){
    float8 a = 1/L;
    float8 b = (2*M_PI*mid)/L;
    float8 y2 = val->y * sin(b*t2) - val->x * cos(b*t2);
    float8 y1 = val->y * sin(b*t1) - val->x * cos(b*t1);
    res = a * (1/b) * (y2-y1);
  } else {
    float8 a = 2/L;
    float8 b = (2*M_PI*N)/L;
    float8 y2 = val->y * sin(b*t2) - val->x * cos(b*t2);
    float8 y1 = val->y * sin(b*t1) - val->x * cos(b*t1);
    res = a * (1/b) * (y2-y1);
  }
  return res;
}

Datum complex_avg_accum(PG_FUNCTION_ARGS) {
  ArrayType* transarray = PG_GETARG_ARRAYTYPE_P(0);
  Complex* new_val = (Complex*) PG_GETARG_POINTER(1);
  int32 t1 = PG_GETARG_INT32(2);
  int32 t2 = PG_GETARG_INT32(3);
  int32 signal_len = PG_GETARG_INT32(4);
  
  float8* transvalues;
  float8 N, Sx, Sy, period;

  transvalues = check_float8_array(transarray, "complex_accum", 4);
  N = transvalues[0];
  Sx = transvalues[1];
  Sy = transvalues[2];
  period = transvalues[3];
  
  Sx += avg_integral_real(N, signal_len, new_val, t1, t2);
  Sy += avg_integral_imag(N, signal_len, new_val, t1, t2);
  N += 1.0;
  period = t2-t1;
  
  if (AggCheckCallContext(fcinfo, NULL)) {
    transvalues[0] = N;
    transvalues[1] = Sx;
    transvalues[2] = Sy;
    transvalues[3] = period;
    
    PG_RETURN_ARRAYTYPE_P(transarray);

  } else {
    Datum transdatums[3];
    ArrayType* result;
  
    transdatums[0] = Float8GetDatumFast(N);
    transdatums[1] = Float8GetDatumFast(Sx);
    transdatums[2] = Float8GetDatumFast(Sy);
    transdatums[3] = Float8GetDatumFast(period);
  
    result = construct_array(transdatums, 4, FLOAT8OID, sizeof(float8), FLOAT8PASSBYVAL, TYPALIGN_DOUBLE);
  
    PG_RETURN_ARRAYTYPE_P(result);
  }
}

Datum complex_avg(PG_FUNCTION_ARGS) {
  ArrayType *transarray = PG_GETARG_ARRAYTYPE_P(0);
  float8 *transvalues;
  float8 Sx, Sy, period;
  
  transvalues = check_float8_array(transarray, "complex_avg", 4);
  Sx = transvalues[1];
  Sy = transvalues[2];
  period = transvalues[3];
  
  /* SQL defines AVG of no values to be NULL */
  if (period == 0.0)
    PG_RETURN_NULL();
  
  Complex* result;
  result = (Complex*)palloc(sizeof(Complex));
  result->x = Sx/period;
  result->y = Sy/period;
  PG_RETURN_POINTER(result);
}

