// test.c - Example program exercising different instruction types
#include <stdio.h>

float add_and_multiply(float a, float b, int x, int y) {
    // Integer operations
    int int_result = x + y;
    int_result *= 2;

    // Float operations
    float float_result = a + b;
    float_result *= 1.5f;

    // Conditionals and comparisons
    if (int_result > 10) {
        float_result += 1.0f;
    }

    // Memory operations through array
    float array[8];
    for (int i = 0; i < 8; i++) {
        array[i] = float_result * i;
    }

    return array[int_result % 8];
}

int main() {
    float result = add_and_multiply(2.5f, 3.5f, 5, 7);
    printf("Result: %f\n", result);
    return 0;
}