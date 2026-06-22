#include <stdio.h>
#include <math.h>
#include "src/turbo-rotation-data.h"

int main() {
    int non_ortho = 0;
    for(int i=0; i<128; i++) {
        for(int j=0; j<128; j++) {
            float sum = 0.0f;
            for(int k=0; k<128; k++) {
                sum += TURBO_ROTATION_R[i*128 + k] * TURBO_ROTATION_R[j*128 + k];
            }
            float expected = (i == j) ? 1.0f : 0.0f;
            if(fabs(sum - expected) > 1e-4) {
                non_ortho++;
            }
        }
    }
    printf("Non-ortho elements: %d\n", non_ortho);
    return 0;
}
