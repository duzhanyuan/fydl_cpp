DEBUG = 0
CXX = g++
ifneq ($(DEBUG),1)
	CFLAGS = -Wall -W -fpic
else
	CFLAGS = -DDEBUG -g -Wall -W -fpic
endif


SRC_PATH = ./src/
FYDL_LIB_OUT_PATH = ./lib/
FYDL_LIB_INC_PATH = ./include/

FYDL_LIB_OUT = $(FYDL_LIB_OUT_PATH)/libfydl.a


FYDL_OBJS = $(SRC_PATH)/TypeDefs.o
FYDL_OBJS += $(SRC_PATH)/Utility.o
FYDL_OBJS += $(SRC_PATH)/Activation.o
FYDL_OBJS += $(SRC_PATH)/StringArray.o
FYDL_OBJS += $(SRC_PATH)/Pattern.o
FYDL_OBJS += $(SRC_PATH)/Timer.o
FYDL_OBJS += $(SRC_PATH)/Matrix.o
FYDL_OBJS += $(SRC_PATH)/MLP_NeuralNetwork.o
FYDL_OBJS += $(SRC_PATH)/Perceptron.o

TEST_SRC_PATH = ./example/
TESTS = mlp_example
TESTS += perceptron_example


.PHONY: all

all: $(FYDL_LIB_OUT) $(TESTS)

$(FYDL_LIB_OUT): $(FYDL_OBJS)
	mkdir -p $(FYDL_LIB_OUT_PATH)
	ar -rv $@ $(FYDL_OBJS)
	mkdir -p $(FYDL_LIB_INC_PATH)
	cp $(SRC_PATH)/*.h $(FYDL_LIB_INC_PATH)
	
	
mlp_example: $(TEST_SRC_PATH)/mlp_example.o
	$(CXX) $(CFLAGS) -o $@ $< -L$(FYDL_LIB_OUT_PATH) -lfydl


perceptron_example: $(TEST_SRC_PATH)/perceptron_example.o
	$(CXX) $(CFLAGS) -o $@ $< -L$(FYDL_LIB_OUT_PATH) -lfydl


.SUFFIXES: .o .cpp .h

.cpp.o:
	$(CXX) $(CFLAGS) -o $@ -c $< -I$(FYDL_LIB_INC_PATH)


.PHONY: clean

clean: 
	rm -rf $(FYDL_LIB_OUT_PATH)
	rm -rf $(FYDL_LIB_INC_PATH)
	rm -rf $(FYDL_OBJS)
	rm -rf $(TEST_SRC_PATH)/*.o
	rm -rf *.o
	rm -rf $(TESTS)

