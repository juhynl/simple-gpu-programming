#pragma once
// /******************************************************************************************************/
// #include <Windows.h>
// static __int64 _start, _freq, _end;
// static float _compute_time;
// #define CHECK_TIME_START(start, freq)                  \
//     QueryPerformanceFrequency((LARGE_INTEGER *)&freq); \
//     QueryPerformanceCounter((LARGE_INTEGER *)&start)
// #define CHECK_TIME_END(start, end, freq, time)      \
//     QueryPerformanceCounter((LARGE_INTEGER *)&end); \
//     time = (float)((float)(end - start) / (freq * 1.0e-3f))
// /******************************************************************************************************/

/******************************************************************************************************/
#include <time.h>

static struct timespec _start, _end;
static float _compute_time;

#define CHECK_TIME_START(start) clock_gettime(CLOCK_MONOTONIC, &start)
#define CHECK_TIME_END(start, end, time)          \
    clock_gettime(CLOCK_MONOTONIC, &end);         \
    time = ((end.tv_sec - start.tv_sec) * 1e3f) + \
           ((end.tv_nsec - start.tv_nsec) / 1e6f)
/******************************************************************************************************/

// CHECK_TIME_START(_start, _freq);
// CHECK_TIME_END(_start, _end, _freq, _compute_time);