# Use the same base image as your current PostgreSQL
FROM postgis/postgis:16-3.4-alpine
# FROM alpine:latest

# Install build dependencies
RUN apk add --no-cache gcc libc-dev postgresql-dev

# Copy the extension source files into the container
COPY src/complex.c app/
COPY src/complex.h app/
COPY src/avg.c app/
COPY src/avg.h app/
COPY src/add.c app/
COPY src/add.h app/

# Set the working directory
WORKDIR /app/

# # Compile and install the 'complex' extension
RUN gcc -c -fPIC -Wall -Werror -g3 -O0 -I/usr/include/postgresql/15/server -o complex.o complex.c && \
    gcc -shared -o complex.so complex.o && \
    cp complex.so $(pg_config --pkglibdir)

# Compile and install the 'avg' extension
RUN gcc -c -fPIC -Wall -Werror -g3 -O0 -I/usr/include/postgresql/15/server -o avg.o avg.c && \
    gcc -shared -o avg.so avg.o && \
    cp avg.so $(pg_config --pkglibdir)
