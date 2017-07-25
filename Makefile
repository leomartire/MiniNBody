CC = gcc
FLAGS = -g -Wall -O3 -lm -fopenmp
CFLAGS = $(FLAGS) -c
SOURCE = MiniNBody
EXEC = mnbody
NAME = MiniNBody

$(EXEC): $(SOURCE).o
	$(CC) $(FLAGS) $(SOURCE).o -o $(EXEC)

$(SOURCE).o: $(SOURCE).c
	$(CC) $(CFLAGS) $(SOURCE).c
  
clean:
	/bin/rm -rf $(EXEC) $(SOURCE).o

tar:
	tar -cvf $(NAME).tar $(SOURCE).c Makefile ReadMe.txt solarSystem.in $(EXEC) *.pdf *.py