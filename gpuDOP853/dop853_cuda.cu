#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Automatically converts std::vector to Python lists
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <tuple>
#include <stdexcept>

// 2. Fixed-size POD struct instead of std::vector
struct State2D {
    double z;
    double v;
};

struct SystemParams {
    double A; // Acoustic Force Amplitude
    double k; // Wavenumber
    double gEf; // Effective Gravity
    double B; // Damping coefficient
    double zEq; // Equilibrium position
    double Lambda; // Wavelength
};

// 3. System RHS - Acoustic Levitator
__device__ void system_rhs(
    double t, 
    const State2D& y, // Estado Atual y(t)
    State2D& dydt,  // Derivadas Atuais y'(t)
    const SystemParams& params) {
    double AmpEf = params.A * params.gEf;
    dydt.z = y.v;
    dydt.v = AmpEf * cos(2.0 * params.k * y.z) - params.gEf - params.B * y.v;
}

// 4. Fixed Butcher Tableau Constants
namespace DOP853Const {
    constexpr int STAGES = 12;
    __constant__ double C[STAGES] = {
        0.0,
        0.526001519587677318785587544488e-01,
        0.789002279381515978178381316732e-01,
        0.118350341907227396726757197510,
        0.281649658092772603273242802490,
        0.333333333333333333333333333333,
        0.25,
        0.307692307692307692307692307692,
        0.651282051282051282051282051282,
        0.6,
        0.857142857142857142857142857142,
        1.0,
    };
    __constant__ double B[STAGES] = {
        5.42937341165687622380535766363e-2,
        0.0,
        0.0,
        0.0,
        0.0,
        4.45031289275240888144113950566,
        1.89151789931450038304281599044,
        -5.8012039600105847814672114227,
        3.1116436695781989440891606237e-1,
        -1.52160949662516078556178806805e-1,
        2.01365400804030348374776537501e-1,
        4.47106157277725905176885569043e-2,
    };
    __constant__ double ER[STAGES] = {
        0.1312004499419488073250102996e-01,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.1225156446376204440720569753e+01,
        -0.4957589496572501915214079952,
        0.1664377182454986536961530415e+01,
        -0.3503288487499736816886487290,
        0.3341791187130174790297318841,
        0.8192320648511571246570742613e-01,
        -0.2235530786388629525884427845e-01,
    };
    __constant__ double A[STAGES][STAGES] = {
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {5.26001519587677318785587544488e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.97250569845378994544595329183e-2, 5.91751709536136983633785987549e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {2.95875854768068491816892993775e-2, 0.0, 8.87627564304205475450678981324e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {2.41365134159266685502369798665e-1, 0.0, -8.84549479328286085344864962717e-1, 9.24834003261792003115737966543e-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.7037037037037037037037037037e-2, 0.0, 0.0, 1.70828608729473871279604482173e-1, 1.25467687566822425016691814123e-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.7109375e-2, 0.0, 0.0, 1.70252211019544039314978060272e-1, 6.02165389804559606850219397283e-2, -1.7578125e-2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {3.70920001185047927108779319836e-2, 0.0, 0.0, 1.70383925712239993810214054705e-1, 1.07262030446373284651809199168e-1, -1.53194377486244017527936158236e-2, 8.27378916381402288758473766002e-3, 0.0, 0.0, 0.0, 0.0, 0.0},
        {6.24110958716075717114429577812e-1, 0.0, 0.0, -3.36089262944694129406857109825, -8.68219346841726006818189891453e-1, 2.75920996994467083049415600797e+1, 2.01540675504778934086186788979e+1, -4.34898841810699588477366255144e+1, 0.0, 0.0, 0.0, 0.0},
        {4.77662536438264365890433908527e-1, 0.0, 0.0, -2.48811461997166764192642586468, -5.90290826836842996371446475743e-1, 2.12300514481811942347288949897e+1, 1.52792336328824235832596922938e+1, -3.32882109689848629194453265587e+1, -2.03312017085086261358222928593e-2, 0.0, 0.0, 0.0},
        {-9.3714243008598732571704021658e-1, 0.0, 0.0, 5.18637242884406370830023853209, 1.09143734899672957818500254654, -8.14978701074692612513997267357, -1.85200656599969598641566180701e+1, 2.27394870993505042818970056734e+1, 2.49360555267965238987089396762, -3.0467644718982195003823669022, 0.0, 0.0},
        {2.27331014751653820792359768449, 0.0, 0.0, -1.05344954667372501984066689879e+1, -2.00087205822486249909675718444, -1.79589318631187989172765950534e+1, 2.79488845294199600508499808837e+1, -2.85899827713502369474065508674, -8.87285693353062954433549289258, 1.23605671757943030647266201528e+1, 6.43392746015763530355970484046e-1, 0.0},
    };
}

// 5. Core Step Function (Stack allocated, no std::vector)
__device__ bool dop853_step(
    double& t, State2D& y, 
    double& h, // Time step
    double rtol, double atol, // Error tolerances
    const SystemParams& params) {
        State2D k[DOP853Const::STAGES]; // Array with STAGES (12 for DOP853) states to perform step
        State2D y_temp; // Auxiliar state

        // Stage 1
        system_rhs(t, y, k[0], params);

        // Stage 2 to 12
        for (int i = 1; i < DOP853Const::STAGES; ++i) {
            y_temp.z = y.z;
            y_temp.v = y.v;

            for (int j = 0; j < i; ++j) {
                y_temp.z += h * DOP853Const::A[i][j] * k[j].z;
                y_temp.v += h * DOP853Const::A[i][j] * k[j].v; 
            }
            system_rhs(t + h * DOP853Const::C[i], y_temp, k[i], params);
        }

        // Combine stages for candidate solution and error estimation
        State2D y_next = y;
        State2D error = {0.0, 0.0};

        for (int i = 0; i < DOP853Const::STAGES; ++i) {
            y_next.z += h * DOP853Const::B[i] * k[i].z;
            y_next.v += h * DOP853Const::B[i] * k[i].v;

            error.z += h * DOP853Const::ER[i] * k[i].z;
            error.v += h * DOP853Const::ER[i] * k[i].v;
        }

        // Adaptive step size control (using math functions compatible with CUDA)
        double scale_z = atol + rtol * fmax(fabs(y.z), fabs(y_next.z));
        double scale_v = atol + rtol * fmax(fabs(y.v), fabs(y_next.v));

        double err_z = fabs(error.z) / scale_z;
        double err_v = fabs(error.v) / scale_v;

        // RMS error across the 2 dimensions
        double max_err = sqrt(0.5 * (err_z * err_z + err_v * err_v));

        if (max_err <= 1.0) {
            // Accept Step
            t += h;
            y = y_next;
        }

        // Compute next step size
        double factor = 0.9 * pow(1.0 / fmax(max_err, 1e-10), 1.0/8.0);
        factor = fmin(5.0, fmax(0.1, factor)); // Clamp factor
        h *= factor;

        return (max_err <= 1.0);
}

__device__ double cubic_spline(
    double s, double h, 
    double x1, double dx1,
    double x2, double dx2){
    double s2 = s * s;
    double s3 = s2 * s;
    double spline = (2.0 * s3 - 3.0 * s2 + 1.0) * x1 +
                    (s3 - 2.0 * s2 + s) * h * dx1 +
                    (-2.0 * s3 + 3.0 * s2) * x2 + 
                    (s3 - s2) * h * dx2;
    return spline;
}

__device__ double cubic_spline_derivative(
    double s, double h, 
    double x1, double dx1,
    double x2, double dx2){
    double s2 = s * s;
    double dx_ds = (6.0 * s2 - 6.0 * s) * x1 + 
                   (3.0 * s2 - 4.0 * s + 1.0) * h * dx1 + 
                   (-6.0 * s2 + 6.0 * s) * x2 + 
                   (3.0 * s2 - 2.0 * s) * h * dx2;
    return dx_ds;
}

// 6. Integration Loop
__global__ void batch_integrate_kernel(
    const State2D* initialConditions, // Array of inputs 
    int numSimulations,
    double* allPeakTimes,             // Flattened output matrix
    double* allPeakPositions,         // Flattened output matrix
    int* allPeakCounts,               // Array of peak counts per thread
    double t0,
    int maxPeaks,
    const double* breakpoints, int num_breakpoints, // Sub_intervals time limits
    const double A0, const double A1, // Amplitude of on/off sub_intervals
    double rtol, double atol, // Error tolerances
    SystemParams params,
    State2D* allBreakpointStates){
        // 1. Thread identifies itself
        int idx = blockIdx.x * blockDim.x + threadIdx.x;    
        if (idx >= numSimulations) return;

        // 2. Load this thread's specific starting state
        State2D y = initialConditions[idx];
        double t = t0;
        int localPeakCount = 0;

        // Perform the integration along each sub_interval
        for (int b = 0; b < num_breakpoints; ++b) {
            double target_t = breakpoints[b];
            params.A = (b % 2 == 0) ? A0 : A1; // Updates the amplitude for this sub_interval

            if (t >= target_t) continue;

            double h = 1e-4; // Initial step guess
            while (t < target_t) {
                if (t + h > target_t) h = target_t - t;
                
                // State before next step
                double t_old = t;
                State2D y_old = y;
                // Size of the next step
                double h_taken = h;
    
                bool accepted = dop853_step(t, y, h, rtol, atol, params);
                if (accepted) {
                    // Verify if v changed sign (z crossed a peak)
                    if (y_old.v * y.v <= 0) {
                        // 1. Get accelerations at the boundaries to build the cubic spline
                        State2D dy_old, dy_new;
                        system_rhs(t_old, y_old, dy_old, params);
                        system_rhs(t, y, dy_new, params);
    
                        double a_old = dy_old.v;
                        double a_new = dy_new.v;
    
                        // 2. Initial guess for the root fraction 's' (Linear Interpolation)
                        double s = (0.0 - y_old.v) / (y.v - y_old.v);
                        
                        // 3. Newton-Raphson Iteration (usually converges to machine precision in 3-4 steps)
                        for (int iter = 0; iter < 10; ++iter) {
    
                            // Evaluate the Cubic Velocity Spline V(s)
                            double v_spline = cubic_spline(s, h_taken, y_old.v, a_old, y.v, a_new);
    
                            // Evaluate the Derivative V'(s) = dV/ds
                            double dv_ds = cubic_spline_derivative(s, h_taken, y_old.v, a_old, y.v, a_new);
    
                            // Break if we hit a flat slope to prevent division by zero
                            if (fabs(dv_ds) < 1e-12) break;
    
                            // Newton-Raphson update
                            double ds = v_spline / dv_ds;
                            s -= ds;
    
                            // Break if converged
                            if (fabs(ds) < 1e-12) break;
                        }
                        // Clamp 's' strictly between 0 and 1 to guarantee it stays inside the step
                        s = fmax(0.0, fmin(1.0, s));
    
                        // 4. Interpolate t_peak
                        double t_peak = t_old + h_taken * s;
    
                        // 5. Interpolate z_peak using the Cubic Position Spline
                        double z_peak = cubic_spline(s, h_taken, y_old.z, y_old.v, y.z, y.v);
    
                        int memOffset = idx * maxPeaks + (localPeakCount % maxPeaks);
                        allPeakTimes[memOffset] = t_peak;
                        allPeakPositions[memOffset] = z_peak;
                        localPeakCount++;
                    }
                    
                    // Verify if the object is too distant from its equilibrium position
                    if (fabs(y.z - params.zEq) - 2 * params.Lambda > 0) {
                        allPeakCounts[idx] = localPeakCount;
                        return; // Terminate Integration
                    }
                }
            }
            int memOffset = idx * num_breakpoints + b;
            allBreakpointStates[memOffset] = y;
        }
        allPeakCounts[idx] = localPeakCount;
    }

namespace py = pybind11;

// This is the wrapper function Python will actually call
std::tuple<
    std::vector<int>,     // Peak counts per simulation
    std::vector<double>,  // Flattened peak times
    std::vector<double>,  // Flattened peak positions
    std::vector<double>,  // Flattened breakpoint z-states
    std::vector<double>   // Flattened breakpoint v-states
>
run_cuda_simulations(
    double t0,
    std::vector<double> z0_list,
    std::vector<double> v0_list,
    std::vector<double> breakpoints,
    double A0, double A1,
    double k, double gEf, double B, double zEq, double Lambda, // System params
    int maxPeaks, double rtol, double atol)
{
    int numSimulations = z0_list.size();
    int numBreakpoints = breakpoints.size();

    if (numSimulations != v0_list.size()) {
        throw std::invalid_argument("z0_list and v0_list must have the same length");
    }
    // 1. Pack initial conditions into State2D structs on the CPU
    std::vector<State2D> host_ics(numSimulations);
    for (int i = 0; i < numSimulations; ++i) {
        host_ics[i] = {z0_list[i], v0_list[i]};
    }

    // 2. Declare GPU memory pointers
    State2D* d_initicalConditionStates;
    double* d_breakpoints;
    double* d_peakTimes;
    double* d_peakPositions;
    int* d_peakCounts;
    State2D* d_breakpointStates;

    // 3. Allocate GPU memory (cudaMalloc)
    cudaMalloc(&d_initicalConditionStates, numSimulations * sizeof(State2D));
    cudaMalloc(&d_breakpoints, numBreakpoints * sizeof(double));
    cudaMalloc(&d_peakTimes, numSimulations * maxPeaks * sizeof(double));
    cudaMalloc(&d_peakPositions, numSimulations * maxPeaks * sizeof(double));
    cudaMalloc(&d_peakCounts, numSimulations * sizeof(int));
    cudaMalloc(&d_breakpointStates, numSimulations * numBreakpoints * sizeof(State2D));

    // 4. Copy data from CPU RAM to GPU VRAM (cudaMemcpy)
    cudaMemcpy(d_initicalConditionStates, host_ics.data(), numSimulations * sizeof(State2D), cudaMemcpyHostToDevice);
    cudaMemcpy(d_breakpoints, breakpoints.data(), numBreakpoints * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_peakCounts, 0, numSimulations * sizeof(int)); // Initialize counts to 0

    // 5. Launch the CUDA Kernel
    SystemParams params = {0.0, k, gEf, B, zEq, Lambda};
    int threadsPerBlock = 256;
    int blocks = (numSimulations + threadsPerBlock - 1) / threadsPerBlock;
    batch_integrate_kernel<<<blocks, threadsPerBlock>>>(
        d_initicalConditionStates, numSimulations, d_peakTimes, d_peakPositions, d_peakCounts,
        t0, maxPeaks, d_breakpoints, numBreakpoints, A0, A1, rtol, atol, params, d_breakpointStates
    );
    
    cudaDeviceSynchronize();

    // 6. Allocate CPU memory to receive results
    std::vector<int> host_peakCounts(numSimulations);
    std::vector<double> host_peakTimes(numSimulations * maxPeaks);
    std::vector<double> host_peakPositions(numSimulations * maxPeaks);
    std::vector<State2D> host_breakpointStates(numSimulations * numBreakpoints);

    // 7. Copy results back from GPU to CPU
    cudaMemcpy(host_peakCounts.data(), d_peakCounts, numSimulations * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_peakTimes.data(), d_peakTimes, numSimulations * maxPeaks * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_peakPositions.data(), d_peakPositions, numSimulations * maxPeaks * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_breakpointStates.data(), d_breakpointStates, numSimulations * numBreakpoints * sizeof(State2D), cudaMemcpyDeviceToHost);

    // 8. Free GPU Memory (Crucial to prevent VRAM leaks)
    cudaFree(d_initicalConditionStates);
    cudaFree(d_breakpoints);
    cudaFree(d_peakTimes);
    cudaFree(d_peakPositions);
    cudaFree(d_peakCounts);
    cudaFree(d_breakpointStates);

    // 9. Split the State2D array into separate z and v arrays for Python
    std::vector<double> breakpoints_z(numSimulations * numBreakpoints);
    std::vector<double> breakpoints_v(numSimulations * numBreakpoints);
    for (int i =0; i < numSimulations * numBreakpoints; ++i) {
        breakpoints_z[i] = host_breakpointStates[i].z;
        breakpoints_v[i] = host_breakpointStates[i].v;
    }

    return std::make_tuple(host_peakCounts, host_peakTimes, host_peakPositions, breakpoints_z, breakpoints_v);
}

// Create the Python module
PYBIND11_MODULE(fast_ode_cuda, m) {
    m.doc() = "C++ DOP853 Integrator for Peak Detection";
    m.def("run_cuda_simulations", &run_cuda_simulations, 
          "Run the ODE solver and return peak times",
          py::arg("t0"), py::arg("z0_list"), py::arg("v0_list"), py::arg("breakpoints"),
          py::arg("A0"), py::arg("A1"), 
          py::arg("k"), py::arg("gEf"), py::arg("B"), py::arg("zEq"), py::arg("Lambda"),
          py::arg("maxPeaks"), py::arg("rtol")=1e-7, py::arg("atol")=1e-10);
}