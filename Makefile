# COMPILER, FLAGS
CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++17
LDFLAGS = -lboost_program_options

# DIRECTORIES
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

SRC_COMMON = $(wildcard $(SRC_DIR)/*.cpp)
SRC_USE = $(filter-out $(SRC_DIR)/arithmeticCode.cpp $(SRC_DIR)/huffman.cpp, $(SRC_COMMON))

OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC_USE))

TARGET = $(BIN_DIR)/bpe

default: $(TARGET)

bpe: $(TARGET)

$(TARGET): $(OBJS)
	mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)


$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

run: $(TARGET)
	./$(TARGET)