# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Produce verbose output by default.
VERBOSE = 1

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64

# Include any dependencies generated for this target.
include programs/CMakeFiles/cmpfillin.dir/depend.make

# Include the progress variables for this target.
include programs/CMakeFiles/cmpfillin.dir/progress.make

# Include the compile flags for this target's objects.
include programs/CMakeFiles/cmpfillin.dir/flags.make

programs/CMakeFiles/cmpfillin.dir/cmpfillin.c.o: programs/CMakeFiles/cmpfillin.dir/flags.make
programs/CMakeFiles/cmpfillin.dir/cmpfillin.c.o: ../../programs/cmpfillin.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object programs/CMakeFiles/cmpfillin.dir/cmpfillin.c.o"
	cd /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64/programs && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cmpfillin.dir/cmpfillin.c.o   -c /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/programs/cmpfillin.c

programs/CMakeFiles/cmpfillin.dir/cmpfillin.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cmpfillin.dir/cmpfillin.c.i"
	cd /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64/programs && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/programs/cmpfillin.c > CMakeFiles/cmpfillin.dir/cmpfillin.c.i

programs/CMakeFiles/cmpfillin.dir/cmpfillin.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cmpfillin.dir/cmpfillin.c.s"
	cd /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64/programs && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/programs/cmpfillin.c -o CMakeFiles/cmpfillin.dir/cmpfillin.c.s

programs/CMakeFiles/cmpfillin.dir/io.c.o: programs/CMakeFiles/cmpfillin.dir/flags.make
programs/CMakeFiles/cmpfillin.dir/io.c.o: ../../programs/io.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object programs/CMakeFiles/cmpfillin.dir/io.c.o"
	cd /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64/programs && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cmpfillin.dir/io.c.o   -c /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/programs/io.c

programs/CMakeFiles/cmpfillin.dir/io.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cmpfillin.dir/io.c.i"
	cd /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64/programs && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/programs/io.c > CMakeFiles/cmpfillin.dir/io.c.i

programs/CMakeFiles/cmpfillin.dir/io.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cmpfillin.dir/io.c.s"
	cd /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64/programs && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/programs/io.c -o CMakeFiles/cmpfillin.dir/io.c.s

programs/CMakeFiles/cmpfillin.dir/smbfactor.c.o: programs/CMakeFiles/cmpfillin.dir/flags.make
programs/CMakeFiles/cmpfillin.dir/smbfactor.c.o: ../../programs/smbfactor.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object programs/CMakeFiles/cmpfillin.dir/smbfactor.c.o"
	cd /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64/programs && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/cmpfillin.dir/smbfactor.c.o   -c /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/programs/smbfactor.c

programs/CMakeFiles/cmpfillin.dir/smbfactor.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/cmpfillin.dir/smbfactor.c.i"
	cd /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64/programs && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/programs/smbfactor.c > CMakeFiles/cmpfillin.dir/smbfactor.c.i

programs/CMakeFiles/cmpfillin.dir/smbfactor.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/cmpfillin.dir/smbfactor.c.s"
	cd /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64/programs && /usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/programs/smbfactor.c -o CMakeFiles/cmpfillin.dir/smbfactor.c.s

# Object files for target cmpfillin
cmpfillin_OBJECTS = \
"CMakeFiles/cmpfillin.dir/cmpfillin.c.o" \
"CMakeFiles/cmpfillin.dir/io.c.o" \
"CMakeFiles/cmpfillin.dir/smbfactor.c.o"

# External object files for target cmpfillin
cmpfillin_EXTERNAL_OBJECTS =

programs/cmpfillin: programs/CMakeFiles/cmpfillin.dir/cmpfillin.c.o
programs/cmpfillin: programs/CMakeFiles/cmpfillin.dir/io.c.o
programs/cmpfillin: programs/CMakeFiles/cmpfillin.dir/smbfactor.c.o
programs/cmpfillin: programs/CMakeFiles/cmpfillin.dir/build.make
programs/cmpfillin: libmetis/libmetis.a
programs/cmpfillin: programs/CMakeFiles/cmpfillin.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C executable cmpfillin"
	cd /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64/programs && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cmpfillin.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
programs/CMakeFiles/cmpfillin.dir/build: programs/cmpfillin

.PHONY : programs/CMakeFiles/cmpfillin.dir/build

programs/CMakeFiles/cmpfillin.dir/clean:
	cd /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64/programs && $(CMAKE_COMMAND) -P CMakeFiles/cmpfillin.dir/cmake_clean.cmake
.PHONY : programs/CMakeFiles/cmpfillin.dir/clean

programs/CMakeFiles/cmpfillin.dir/depend:
	cd /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0 /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/programs /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64 /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64/programs /mnt/c/Users/chaoq/Documents/GitHub/penguin_partitioning/metis-5.1.0/build/Linux-x86_64/programs/CMakeFiles/cmpfillin.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : programs/CMakeFiles/cmpfillin.dir/depend

