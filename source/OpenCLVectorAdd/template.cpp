#include <wb.h>
#include <CL/opencl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

//@@ Write the OpenCL kernel
//const char *kernelSource =	"first_line_of_code \n"
//								"next_line_of_code  \n"
//								...
//								"last_line_of_code  \n";
const char* kernalSource;

int main(int argc, char *argv[]) {
  wbArg_t args;
  int inputLength;
  int inputLengthBytes;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  cl_mem deviceInput1;
  cl_mem deviceInput2;
  cl_mem deviceOutput;
  deviceInput1 = NULL;
  deviceInput2 = NULL;
  deviceOutput = NULL;

  cl_platform_id cpPlatform; // OpenCL platform
  cl_device_id device_id;    // device ID
  cl_context context;        // context
  cl_command_queue queue;    // command queue
  cl_program program;        // program
  cl_kernel kernel;          // kernel

  context = NULL;        
  queue = NULL;    
  program = NULL;
  kernel = NULL;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  inputLengthBytes = inputLength * sizeof(float);
  hostOutput       = (float *)malloc(inputLengthBytes);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The input length is ", inputLength);
  wbLog(TRACE, "The input size is ", inputLengthBytes, " bytes");

  //@@ Initialize the workgroup dimensions
  cl_int clerr = CL_SUCCESS;
  cl_platform_id cpPlatform;    // OpenCL platform
  cl_device_id device_id;       // device ID

  //@@ Bind to platform
  clerr = clGetPlatformIDs(1, &cpPlatform, NULL);

  //@@ Get ID for the device
  clerr = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

  //@@ Create a context
  cl_context clctx = clCreateContext(0, 1, &device_id, NULL, NULL, &clerr);

  //@@ Create a command queue
  cl_command_queue clcmdq = clCreateCommandQueue(clctx, device_id, 0, &clerr);

  //@@ Create the compute program from the source buffer
  cl_program clpgm;
  clpgm = clCreateProgramWithSource(clctx, 1, &kernalSource, NULL, &clerr);

  //@@ Build the program executable
  clerr = clBuildProgram(clpgm, 0, NULL, NULL, NULL, NULL);

  //@@ Create the compute kernel in the program we wish to run
  cl_kernel clkern = clCreateKernel(clpgm, "kernelSource", &clerr);

  //@@ Create the input and output arrays in device memory for our calculation
  //@@ Write our data set into the input array in device memory
  deviceInput1 = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputLengthBytes, hostInput1, NULL);
  deviceInput2 = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, inputLengthBytes, hostInput2, NULL);
  deviceOutput = clCreateBuffer(clctx, CL_MEM_WRITE_ONLY, inputLengthBytes, NULL, NULL);

  //@@ Set the arguments to our compute kernel
  clerr = clSetKernelArg(clkern, 0, sizeof(cl_mem), (void *)&deviceInput1);
  clerr = clSetKernelArg(clkern, 1, sizeof(cl_mem), (void*)&deviceInput2);
  clerr = clSetKernelArg(clkern, 2, sizeof(cl_mem), (void*)&deviceOutput);
  clerr = clSetKernelArg(clkern, 3, sizeof(int), &inputLength);

  //@@ Execute the kernel over the entire range of the data set
  cl_event event = NULL;
  const size_t Gsz = ((inputLength-1) / 256 + 1) * 256;
  const size_t Bsz = 256;
  clerr = clEnqueueNDRangeKernel(clcmdq, clkern, 1, NULL, &Gsz, &Bsz, 0, NULL, &event);

  //@@ Wait for the command queue to get serviced before reading back results
  clerr = clWaitForEvents(1, &event);

  //@@ Read the results from the device
  clEnqueueReadBuffer(clcmdq, deviceOutput, CL_TRUE, 0, inputLengthBytes, hostOutput, 0, NULL, NULL);

  wbSolution(args, hostOutput, inputLength);

  // release OpenCL resources
  clReleaseMemObject(deviceInput1);
  clReleaseMemObject(deviceInput2);
  clReleaseMemObject(deviceOutput);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  // release host memory
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);

  return 0;
}
