# Makefile for ParticleNet C++ libraries

CXX = g++
ROOTFLAGS = $(shell root-config --cflags)
ROOTLIBS = $(shell root-config --libs)
CXXFLAGS = -std=c++17 -O3 -fPIC -Iinclude $(ROOTFLAGS)
LDFLAGS = $(ROOTLIBS)

# Directories
SRCDIR = src
OBJDIR = obj
LIBDIR = libs
INCDIR = include
TESTDIR = test

# Source files
SOURCES = $(wildcard $(SRCDIR)/*.cc)
OBJECTS = $(patsubst $(SRCDIR)/%.cc,$(OBJDIR)/%.o,$(SOURCES))
LIBRARY = $(LIBDIR)/libParticleNet.so

# Test program
TEST_PROG = test_dataformat

# Default target
all: dirs $(LIBRARY)

# Build everything including test
test: all $(TEST_PROG)

# Create necessary directories
dirs:
	@mkdir -p $(OBJDIR) $(LIBDIR)

# Build shared library
$(LIBRARY): $(OBJECTS)
	$(CXX) -shared -o $@ $^ $(LDFLAGS)
	@echo "Built library: $@"

# Compile object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cc $(INCDIR)/%.h
	$(CXX) $(CXXFLAGS) -c $< -o $@
	@echo "Compiled: $<"

# Build test program
$(TEST_PROG): $(TESTDIR)/test_dataformat.cpp $(LIBRARY)
	$(CXX) $(CXXFLAGS) -o $(TESTDIR)/$@ $< -L$(LIBDIR) -lParticleNet $(LDFLAGS)
	@echo "Built test program: $(TESTDIR)/$@"

# Clean up
clean:
	rm -rf $(OBJDIR) $(LIBDIR) $(TESTDIR)/$(TEST_PROG)

# Install library (already in libs directory after build)
install: all
	@echo "Library installed in $(LIBDIR)/"

.PHONY: all clean install dirs test
