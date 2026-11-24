/* Run with the Ventus OpenCL ICD and VENTUS_BACKEND=rtlsim.
 * cc -std=c11 -O2 opencl_global_amo.c -lOpenCL -o opencl_global_amo
 */
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { lanes = 128, banks = 4 };
static cl_context context;
static cl_command_queue queue;
static cl_program program;

static void check(cl_int status, const char *operation) {
    if (status != CL_SUCCESS) {
        fprintf(stderr, "%s failed (%d)\n", operation, status);
        exit(EXIT_FAILURE);
    }
}
#define CHECK(call) check((call), #call)

static const char source[] =
    "#define COUNT_KERNEL(name, op) \\\n"
    "kernel void name(global int *counter, global int *out) { \\\n"
    "  uint i = get_global_id(0); \\\n"
    "  if (i % 3 != 0) out[i] = op; \\\n"
    "}\n"
    "COUNT_KERNEL(seqcst, atomic_add(counter + i % 4, 1))\n"
    "COUNT_KERNEL(relaxed, __atomic_fetch_add(counter + i % 4, 1, __ATOMIC_RELAXED))\n"
    "COUNT_KERNEL(acquire, __atomic_fetch_add(counter + i % 4, 1, __ATOMIC_ACQUIRE))\n"
    "COUNT_KERNEL(release, __atomic_fetch_add(counter + i % 4, 1, __ATOMIC_RELEASE))\n"
    "kernel void cached(global uint *words, global uint *out) {\n"
    "  uint i = get_global_id(0);\n"
    "  volatile global uint *p = words + i;\n"
    "  uint before = *p;\n"
    "  *p = before + 7;\n"
    "  uint old = __atomic_fetch_add(p, 5u, __ATOMIC_RELAXED);\n"
    "  uint after = *p;\n"
    "  out[i] = (old == before + 7 && after == before + 12);\n"
    "}\n"
    "kernel void operations(global uint *words, global uint *out) {\n"
    "  uint i = get_global_id(0), value = i + 9;\n"
    "  global uint *p = words + i;\n"
    "  switch (i % 9) {\n"
    "    case 0: out[i] = atomic_add(p, value); break;\n"
    "    case 1: out[i] = atomic_xchg(p, value); break;\n"
    "    case 2: out[i] = atomic_and(p, value); break;\n"
    "    case 3: out[i] = atomic_or(p, value); break;\n"
    "    case 4: out[i] = atomic_xor(p, value); break;\n"
    "    case 5: out[i] = atomic_min((global int *)p, (int)value); break;\n"
    "    case 6: out[i] = atomic_max((global int *)p, (int)value); break;\n"
    "    case 7: out[i] = atomic_min(p, value); break;\n"
    "    case 8: out[i] = atomic_max(p, value); break;\n"
    "  }\n"
    "}\n";

static void dispatch(const char *name, uint32_t *words, size_t count, uint32_t *out) {
    cl_int status;
    cl_kernel kernel = clCreateKernel(program, name, &status);
    check(status, "clCreateKernel");
    size_t sizes[] = { count * sizeof(*words), lanes * sizeof(*out) };
    void *data[] = { words, out };
    cl_mem buffers[2];
    for (cl_uint i = 0; i < 2; ++i) {
        buffers[i] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    sizes[i], data[i], &status);
        check(status, "clCreateBuffer");
        CHECK(clSetKernelArg(kernel, i, sizeof(cl_mem), &buffers[i]));
    }
    size_t global = lanes, local = 32;
    CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL));
    for (cl_uint i = 0; i < 2; ++i) {
        CHECK(clEnqueueReadBuffer(queue, buffers[i], CL_TRUE, 0, sizes[i], data[i], 0, NULL, NULL));
        CHECK(clReleaseMemObject(buffers[i]));
    }
    CHECK(clReleaseKernel(kernel));
}

static int check_counts(const char *name) {
    uint32_t counters[banks] = {0}, out[lanes], expected[banks] = {0};
    unsigned char seen[banks][lanes] = {{0}};
    memset(out, 0xff, sizeof(out));
    for (unsigned i = 0; i < lanes; ++i)
        if (i % 3) ++expected[i % banks];
    dispatch(name, counters, banks, out);
    for (unsigned i = 0; i < lanes; ++i) {
        unsigned bank = i % banks;
        if (i % 3 == 0) {
            if (out[i] == UINT32_MAX) continue;
        } else if (out[i] < expected[bank] && !seen[bank][out[i]]) {
            seen[bank][out[i]] = 1;
            continue;
        }
        fprintf(stderr, "%s: invalid return at lane %u: %u\n", name, i, out[i]);
        return 1;
    }
    for (unsigned j = 0; j < banks; ++j) {
        if (counters[j] == expected[j]) continue;
        fprintf(stderr, "%s: counter %u: got %u, expected %u\n", name, j, counters[j], expected[j]);
        return 1;
    }
    return 0;
}

static uint32_t initial(unsigned i) { return UINT32_C(0x80000000) + i; }
static uint32_t updated(unsigned i) {
    uint32_t old = initial(i), arg = i + 9;
    switch (i % 9) {
        case 0: return old + arg;
        case 1: return arg;
        case 2: return old & arg;
        case 3: return old | arg;
        case 4: return old ^ arg;
        case 5: case 8: return old;
        default: return arg;
    }
}

static int check_operations(void) {
    uint32_t words[lanes], out[lanes] = {0};
    for (unsigned i = 0; i < lanes; ++i) words[i] = initial(i);
    dispatch("operations", words, lanes, out);
    for (unsigned i = 0; i < lanes; ++i) {
        if (out[i] == initial(i) && words[i] == updated(i)) continue;
        fprintf(stderr, "lane %u: returned %08x, stored %08x (expected %08x, %08x)\n",
                i, out[i], words[i], initial(i), updated(i));
        return 1;
    }
    return 0;
}

static int check_cached(void) {
    uint32_t words[lanes], out[lanes] = {0};
    for (unsigned i = 0; i < lanes; ++i) words[i] = initial(i);
    dispatch("cached", words, lanes, out);
    for (unsigned i = 0; i < lanes; ++i) {
        if (out[i] == 1 && words[i] == initial(i) + 12) continue;
        fprintf(stderr, "cached: incorrect load/store or AMO at lane %u, word %08x\n", i, words[i]);
        return 1;
    }
    return 0;
}

int main(void) {
    cl_platform_id platform;
    cl_device_id device;
    cl_int status;
    CHECK(clGetPlatformIDs(1, &platform, NULL));
    CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL));
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    check(status, "clCreateContext");
    queue = clCreateCommandQueue(context, device, 0, &status);
    check(status, "clCreateCommandQueue");
    const char *code = source;
    program = clCreateProgramWithSource(context, 1, &code, NULL, &status);
    check(status, "clCreateProgramWithSource");
    status = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (status != CL_SUCCESS) {
        char log[8192] = {0};
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
        fprintf(stderr, "%s\n", log);
        check(status, "clBuildProgram");
    }
    const char *orders[] = {"relaxed", "acquire", "release", "seqcst"};
    int failed = 0;
    for (unsigned i = 0; i < sizeof(orders)/sizeof(orders[0]); ++i)
        failed |= check_counts(orders[i]);
    failed |= check_operations();
    failed |= check_cached();
    CHECK(clReleaseProgram(program));
    CHECK(clReleaseCommandQueue(queue));
    CHECK(clReleaseContext(context));
    if (!failed) puts("Global word AMOs passed");
    return failed;
}
