
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

#include "complex.h"

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(complex_in);
PG_FUNCTION_INFO_V1(complex_out);
PG_FUNCTION_INFO_V1(complex_recv);
PG_FUNCTION_INFO_V1(complex_send);
PG_FUNCTION_INFO_V1(complex_add);

Datum complex_in(PG_FUNCTION_ARGS) {
  char* str = PG_GETARG_CSTRING(0);
  double x, y;
  Complex* result;

  if (sscanf(str, "(%lf%lfj)", &x, &y) != 2)
      ereport(ERROR,
        (errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
          errmsg("invalid input syntax for type %s: \"%s\"",
            "complex", str)));

  result = (Complex *) palloc(sizeof(Complex));
  result->x = x;
  result->y = y;
  PG_RETURN_POINTER(result);
}
  
Datum complex_out(PG_FUNCTION_ARGS) {
  Complex* complex = (Complex*) PG_GETARG_POINTER(0);
  char* result;

  if (complex->y >= 0.0) {
    result = psprintf("(%g+%gj)", complex->x, complex->y);
  } else {
    result = psprintf("(%g%gj)", complex->x, complex->y);
  }
  PG_RETURN_CSTRING(result);
}

Datum complex_recv(PG_FUNCTION_ARGS) {
  StringInfo buf = (StringInfo) PG_GETARG_POINTER(0);
  Complex* result;

  result = (Complex*) palloc(sizeof(Complex));
  result->x = pq_getmsgfloat8(buf);
  result->y = pq_getmsgfloat8(buf);
  PG_RETURN_POINTER(result);
}

Datum complex_send(PG_FUNCTION_ARGS) {
  Complex* complex = (Complex*) PG_GETARG_POINTER(0);
  StringInfoData buf;

  pq_begintypsend(&buf);
  pq_sendfloat8(&buf, complex->x);
  pq_sendfloat8(&buf, complex->y);
  PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
}
    
Datum complex_add(PG_FUNCTION_ARGS) {
  Complex* a = (Complex*) PG_GETARG_POINTER(0);
  Complex* b = (Complex*) PG_GETARG_POINTER(1);
  Complex* result;
  
  result = (Complex*) palloc(sizeof(Complex));
  result->x = a->x + b->x;
  result->y = a->y + b->y;
  PG_RETURN_POINTER(result);
}