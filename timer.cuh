#pragma once

#include <time.h>
#include <string>

inline clock_t startTimer() {
    return clock();
}

inline void stopTimer(clock_t startTime, std::string funcName) {
    int delta = (float)(clock() - startTime) / CLOCKS_PER_SEC * 1000;
    printf("%s took %d ms\n", funcName.c_str(), delta);
}
