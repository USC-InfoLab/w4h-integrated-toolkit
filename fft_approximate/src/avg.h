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

static float8 avg_integral_real(float8 N, float8 L, Complex* val, int32 t1, int32 t2);
static float8 avg_integral_imag(float8 N, float8 L, Complex* val, int32 t1, int32 t2);
Datum complex_avg_accum(PG_FUNCTION_ARGS);
Datum complex_avg(PG_FUNCTION_ARGS);