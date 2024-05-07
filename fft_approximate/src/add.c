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

#include "add.h"

#define TOPK 100

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(complex_add);

Datum complex_add(PG_FUNCTION_ARGS){
  Complex* num = (Complex*) PG_GETARG_POINTER(0);
  result = (Complex*) palloc(sizeof(Complex));
  result->x = num->x + 1;
  result->y = num->y + 1;
  PG_RETURN_POINTER(result);
}