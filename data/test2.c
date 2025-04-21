
struct Data {       // Struct chunk
    int id;
};

void log_data(struct Data d) {  // Function chunk
    printf("%d", d.id);
}#include <stdio.h>  // Should be a preprocessor chunk
