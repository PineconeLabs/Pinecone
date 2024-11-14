# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -g
SRCDIR = src
TARGET = libpinecone
INCLUDESPATH = -I../include/ # put a .. infront of the actual include path because in hte compiling process it goes into the src/ directory
# Default to static
LIBTYPE = static

# Source files and corresponding object files
SRCS = $(wildcard $(SRCDIR)/*.c) # Look for .c files in the src directory
OBJS = $(SRCS:$(SRCDIR)/%.c=$(SRCDIR)/%.o) # Update to include the SRCDIR

SRCS_NO_PREFIX = $(subst $(SRCDIR)/, , $(SRCS))
OBJS_NO_PREFIX = $(subst $(SRCDIR)/, $(SRCDIR)/, $(OBJS))


OUTPUTDIR ?= build

# Define the target library
TARGET_LIB = $(TARGET).a

# Default target
all: pinecone
.PHONY: all clean

# Rule to compile source files and create the static library
pinecone:
	@echo "Compiling source files..."

	cd $(SRCDIR) && $(CC) $(CFLAGS) -c $(SRCS_NO_PREFIX) $(INCLUDESPATH)


	ar rcs $(TARGET_LIB) $(OBJS_NO_PREFIX)

	mkdir -p $(OUTPUTDIR)

	mv $(TARGET_LIB) $(OUTPUTDIR)
	cp src/pinecone.h $(OUTPUTDIR)
	rm -f $(SRCDIR)/*.o

# Clean target to remove object files and library
clean:
	@echo "Cleaning up..."
	rm -f $(SRCDIR)/*.o
	rm -rf $(OUTPUTDIR)
