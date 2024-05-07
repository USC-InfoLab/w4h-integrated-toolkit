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