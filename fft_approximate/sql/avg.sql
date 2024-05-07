CREATE FUNCTION complex_avg_accum(anyarray, complex, integer, integer, integer) 
  RETURNS anyarray AS '/usr/lib/postgresql/avg.so', 'complex_avg_accum' 
  LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION complex_avg(anyarray) 
  RETURNS complex AS '/usr/lib/postgresql/avg.so', 'complex_avg' 
  LANGUAGE C IMMUTABLE STRICT;

CREATE AGGREGATE approximate_avg(complex, integer, integer, integer) 
  (sfunc = complex_avg_accum, stype = float8[], finalfunc = complex_avg, initcond = '{0,0,0,0}');