
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

typedef struct Complex {
  double x;
  double y;
} Complex;

Datum complex_in(PG_FUNCTION_ARGS);
Datum complex_out(PG_FUNCTION_ARGS);
Datum complex_recv(PG_FUNCTION_ARGS);
Datum complex_send(PG_FUNCTION_ARGS);
Datum complex_add(PG_FUNCTION_ARGS);