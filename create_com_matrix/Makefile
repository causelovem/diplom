OTF2PATH = /opt/otf2

create_comm_matrix: create_comm_matrix.c
	gcc -Wall -Werror -I$(OTF2PATH)/include -L$(OTF2PATH)/lib \
	-Wl,-rpath -Wl,$(OTF2PATH)/lib \
	$^ -lotf2 -lm -o $@

clean:
	rm -f create_comm_matrix
