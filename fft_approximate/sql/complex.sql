CREATE TYPE complex;

CREATE FUNCTION complex_in(cstring) 
  RETURNS complex AS '/usr/lib/postgresql/complex.so', 'complex_in' 
  LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION complex_out(complex) 
  RETURNS cstring AS '/usr/lib/postgresql/complex.so', 'complex_out' 
  LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION complex_recv(internal) 
  RETURNS complex AS '/usr/lib/postgresql/complex.so', 'complex_recv' 
  LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION complex_send(complex) 
  RETURNS bytea AS '/usr/lib/postgresql/complex.so', 'complex_send' 
  LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION complex_add(complex, complex)
  RETURNS complex AS '/usr/lib/postgresql/complex.so', 'complex_add' 
  LANGUAGE C IMMUTABLE STRICT;

CREATE TYPE complex (internallength = 16, 
                      input = complex_in, 
                      output = complex_out, 
                      receive = complex_recv, 
                      send = complex_send, 
                      alignment = double);