#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Automatically converts std::vector to Python lists
#include <pybind11/numpy.h> // Add this
#include <vector>
#include <cmath>
#include <tuple>
#include <algorithm>

// Fixed-size POD struct instead of std::vector
struct State2D {
    double z;
    double v;
};

// Keeps track of the peaks 
struct PeakTracker {
    std::vector<double>* peak_times;
    std::vector<double>* peak_positions;
    int max_peaks; // Max number of peaks tracked 
    int total_peaks; // Tracks the number of peaks found
    double start_time;
};

// Relevant parameters of the System
struct SystemParams {
    double A; // Acoustic Force Amplitude (in multiples of gEf)
    double two_k; // 2 * Wavenumber (Precomputed for the Acoustic force)
    double gEf; // Effective Gravity
    double B; // Damping coefficient
    double phi; // Acoustic Force Phase
    double dA; // Amplitude Modulation Factor
    double omega; // Angular Frequency of Modulation
    double zEq; // Equilibrium position
    double Lambda; // Wavelength
};

// Keeps track of all validated steps of the integration 
struct TrajectoryTracker {
    std::vector<double> t;
    std::vector<double> z;
    std::vector<double> v;
    bool captured = true;
};

// Keeps informations about the integration process
struct Status {
    int nAccepted = 0;
    int nRejected = 0;
    int MODE;
};

// Fixed Butcher Tableau Constants for DOP853
namespace DOP853Const {
    constexpr int STAGES = 12;
    constexpr double C[STAGES] = {
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
    constexpr double B[STAGES] = {
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
    constexpr double ER[STAGES] = {
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
    constexpr double A[STAGES][STAGES] = {
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

// Movement equation of Forced Oscillations in the Acoustic Levitator
template <int MODE>
inline void levitatorODE(
    double t, 
    const State2D& y, // State y(t)
    State2D & dydt, // State y'(t)
    const SystemParams& params){
        dydt.z = y.v;
        double AmpEf; // Effective Amplitude

        if constexpr (MODE == 0 || MODE == 1) {
            // On-Off Intervals (Alternating Amplitude) || Phase Jumps (Alternating Phase)
            AmpEf = params.A * params.gEf;
        }
        else if constexpr (MODE == 2) {
            // Sine Modulation (Amplitude Continuos Modulation)
            double f_t = 1.0 + params.dA * sin(params.omega * t);
            AmpEf = f_t * params.A * params.gEf;
        }
        
        // z'' = A_Ef * cos(2*k*z - phi) - gEf - b * v
        dydt.v = AmpEf * cos(params.two_k * y.z - params.phi) - params.gEf - params.B * y.v;
    }

// Core Step Function
template <int MODE>
bool dop853_step(
    double& t, State2D& y, 
    double& h, // Time step
    double rtol, double atol, // Error tolerances
    const SystemParams& params) {
        State2D k[DOP853Const::STAGES]; // Array with STAGES (12 for DOP853) states to perform step
        State2D y_temp; // Auxiliar state

        // Stage 1
        levitatorODE<MODE>(t, y, k[0], params);

        // Stage 2 to 12
        for (int i = 1; i < DOP853Const::STAGES; ++i) {
            y_temp.z = y.z;
            y_temp.v = y.v;

            for (int j = 0; j < i; ++j) {
                y_temp.z += h * DOP853Const::A[i][j] * k[j].z;
                y_temp.v += h * DOP853Const::A[i][j] * k[j].v; 
            }
            levitatorODE<MODE>(t + h * DOP853Const::C[i], y_temp, k[i], params);
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
            t += h; // Update the current time
            y = y_next; // Update the state
        }

        // Compute next step size
        double factor = 0.9 * pow(1.0 / fmax(max_err, 1e-10), 1.0/8.0);
        factor = fmin(5.0, fmax(0.1, factor)); // Clamp factor
        h *= factor; // Update next step
        
        // Returns if the step was accepted (true) or rejected (false)
        return (max_err <= 1.0);
}

// Cubic Spline interpolator
double cubicSpline(
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

// Derivative of a Cubic Spline interpolator (dx/ds)
double cubicSplineDerivative(
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

// Interpolates the time and position of a peak (v = 0) between two steps
template <int MODE>
void interpolatePeak(
    double tNew, State2D yNew, // State after the step
    double tOld, State2D yOld,  // State before the step
    double hTaken, // Size of the step
    double timeOffSet, // Offset for the current sub-interval 
    SystemParams& params,
    PeakTracker& tracker){
        // Get accelerations at the boundaries to build the cubic spline
        State2D dyOld, dyNew;
        levitatorODE<MODE>(tOld, yOld, dyOld, params);
        levitatorODE<MODE>(tNew, yNew, dyNew, params);
        double aOld = dyOld.v;
        double aNew = dyNew.v;

        // Initial guess for the root fraction 's' (Linear Interpolation)
        double s = (0.0 - yOld.v) / (yNew.v - yOld.v);

        // Newton-Raphson Iteration to determine 's'
        for (int iter = 0; iter < 15; ++iter) {

            // Evaluate the Cubic Velocity Spline v(s)
            double v_spline = cubicSpline(s, hTaken, yOld.v, aOld, yNew.v, aNew);

            // Evaluate the Derivative v'(s) = dv/ds
            double dv_ds = cubicSplineDerivative(s, hTaken, yOld.v, aOld, yNew.v, aNew);

            // Break if it hits a flat slope to prevent division by zero
            if (fabs(dv_ds) < 1e-12) break;

            // Newton-Raphson update
            double ds = v_spline / dv_ds;
            s -= ds;

            // Break if converged
            if (fabs(ds) < 1e-12) break;
        }

        // Clamp 's' strictly between 0 and 1 to guarantee it stays inside the step
        s = fmax(0.0, fmin(1.0, s));

        // Interpolate tPeak with the refined 's'
        double tPeak = tOld + hTaken * s + timeOffSet; // In GLOBAL time

        // Interpolate zPeak using the Cubic Position Spline
        double zPeak = cubicSpline(s, hTaken, yOld.z, yOld.v, yNew.z, yNew.v);

        // Check if the peak should be tracked
        if (tPeak >= tracker.start_time) {
            // If the number of peaks is unbound
            if (tracker.max_peaks < 0) {
                tracker.peak_times -> push_back(tPeak);
                tracker.peak_positions -> push_back(zPeak);
            } else { 
                // If only max_peaks should be tracked
                int index = tracker.total_peaks % tracker.max_peaks;
                (*tracker.peak_times)[index] = tPeak;
                (*tracker.peak_positions)[index] = zPeak;
            }
            tracker.total_peaks++;
        }
}

// Integration Loop
template <int MODE>
void integrate_dop853(
    State2D& y, double& t,
    const double* breakpoints, int num_breakpoints, // Sub-intervals time limits
    const double var0, const double var1, // Amplitude or Phase of on/off sub-intervals
    double rtol, double atol, double initialStep, // Error tolerances and initial step size
    SystemParams& params, // Parameters that defines the system
    PeakTracker& trackerMax, // Tracker of the maxima
    PeakTracker& trackerMin, // Tracker of the minima
    TrajectoryTracker& trajectory, // Tracker of validated steps
    Status& integrationStatus){ 
        // Record the initial state using the GLOBAL initial time
        trajectory.t.push_back(breakpoints[0]);
        trajectory.z.push_back(y.z);
        trajectory.v.push_back(y.v);

        // Perform the integration along each sub-interval
        for (int b = 1; b < num_breakpoints; ++b) {
            // Local size of this sub-interval
            double target_t = breakpoints[b] - breakpoints[b-1];
            if constexpr (MODE == 0) {
                params.A = ((b-1) % 2 == 0) ? var0 : var1; // Updates the AMPLITUDE for this sub_interval
            } else if constexpr (MODE == 1) {
                params.phi = ((b-1) % 2 == 0) ? var0 : var1; // Updates the PHASE for this sub_interval
            }

            if (target_t <= 0) continue;

            double h = initialStep; // Initial step guess

            // Integrate purely in local time [0, target_t]
            while (t < target_t) {
                if (t + h > target_t) h = target_t - t;
                
                // State before next step
                double t_old = t;
                State2D y_old = y;
                // Size of the next step
                double h_taken = h;
    
                bool accepted = dop853_step<MODE>(t, y, h, rtol, atol, params);
                if (accepted) {
                    integrationStatus.nAccepted++;
                    // Determine the offset for this specific sub-interval
                    double current_offset = breakpoints[b-1];
                    // Verify if the step crosses the startTime of the trackers
                    if (t + current_offset >= trackerMax.start_time) { 
                        // Verify if v changed sign (z crossed a peak)
                        if (y_old.v > 0.0 && y.v <= 0) interpolatePeak<MODE>(t, y, t_old, y_old, h_taken, current_offset, params, trackerMax);
                        else if (y_old.v < 0.0 && y.v >= 0) interpolatePeak<MODE>(t, y, t_old, y_old, h_taken, current_offset, params, trackerMin);
                    }
                    // Add the accepted step to the tracker
                    trajectory.t.push_back(t + current_offset);
                    trajectory.z.push_back(y.z);
                    trajectory.v.push_back(y.v);
                    
                    // Verify if the object is too distant from its equilibrium position (Levitator unable to capture it)
                    if (fabs(y.z - params.zEq) - 2 * params.Lambda > 0) {
                        trajectory.captured = false; // Mark the trajectory as not captured
                        return; // Terminate Integration
                    }
                }
                else integrationStatus.nRejected++;
            }
            // Reset local time to 0 for the start of the next sub-interval
            t = 0.0;
        }
    }

// Organize the peaks in cronological order if a fixed number was being tracked
void readPeaksTracker(
    PeakTracker& tracker, 
    std::vector<double>& peakTimes, std::vector<double>& peakPositions) {
        int peaks_to_read = (tracker.total_peaks < tracker.max_peaks) ? tracker.total_peaks : tracker.max_peaks;
        int start_idx = (tracker.total_peaks < tracker.max_peaks) ? 0 : (tracker.total_peaks % tracker.max_peaks);
        for (int i = 0; i < peaks_to_read; ++i) {
            int idx = (start_idx + i) % tracker.max_peaks;
            peakTimes.push_back( (*tracker.peak_times)[idx] );
            peakPositions.push_back( (*tracker.peak_positions)[idx] );
        }
    }

namespace py = pybind11;

template <typename T>
// Helper function to convert std::vector<T> to py::array_t<T>, allowing to return NumPy arrays directly from C++ without copying the data.
py::array_t<T> as_pyarray(std::vector<T>&& vec) {
    // Move the vector to the heap so it survives after the function returns
    auto* ptr = new std::vector<T>(std::move(vec));

    // Create a Python capsule to delete the vector later
    auto capsule = py::capsule(ptr, [](void* p) {
        delete reinterpret_cast<std::vector<T>*>(p);
    });

    // Return the NumPy array pointing directly to the C++ memory
    return py::array_t<T>(ptr->size(), ptr->data(), capsule);
}

// Wrapper function Python will actually call
std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>, py::array_t<double>, 
           py::array_t<double>, py::array_t<double>, py::array_t<double>, 
           std::tuple<bool, int, int, int>>
run_simulation(double z0, double v0, 
               std::vector<double> breakpoints,
               double k, double gEf, double B, double zEq, double Lambda, // System params
               double val0, double val1,
               double base_A, double base_phi, double dA, double omega,
               int max_peaks, double trackerStartTime, int simMode,
               double rtol, double atol, double initialStep)
{
    // Setup the inital state
    State2D y = {z0, v0};
    // Start local integration time at 0
    double t = 0.0;
    // Sets the initial system parameters
    SystemParams params = {base_A, 2*k, gEf, B, base_phi, dA, omega, zEq, Lambda};

    double totalIntegrationTime = breakpoints.back() - breakpoints.front();

    // Estimate total steps with 10 steps per time unit and a minimum of 1000
    int estimatedSteps = std::max(1000, static_cast<int>(10.0 * totalIntegrationTime));

    // Creates the steps tracker
    TrajectoryTracker trajectory;
    // Pre-allocate memory to minimize dynamic reallocation 
    trajectory.t.reserve(estimatedSteps); 
    trajectory.z.reserve(estimatedSteps);
    trajectory.v.reserve(estimatedSteps);

    // Dynamically allocate memory for the number of peaks requested
    int initialSize = (max_peaks > 0) ? max_peaks : 0;
    std::vector<double> max_t_mem(initialSize, 0.0);
    std::vector<double> max_z_mem(initialSize, 0.0);
    std::vector<double> min_t_mem(initialSize, 0.0);
    std::vector<double> min_z_mem(initialSize, 0.0);

    // Point the tracker at this memory
    PeakTracker trackerMax = {&max_t_mem, &max_z_mem, max_peaks, 0, trackerStartTime};
    PeakTracker trackerMin = {&min_t_mem, &min_z_mem, max_peaks, 0, trackerStartTime};

    // Get the raw pointer and size to pass to the core function
    const double* bp_ptr = breakpoints.data();
    // The number of breakpoints never reach billions
    int num_bp = static_cast<int>(breakpoints.size());

    Status integrationStatus = {0,0,simMode};

    // Perform the integration
    if (simMode == 0) {
        params.A = val0;
        integrate_dop853<0>(y, t, bp_ptr, num_bp, val0, val1, rtol, atol, initialStep, params, trackerMax, trackerMin, trajectory, integrationStatus);
    } else if (simMode == 1) {
        params.phi = val0;
        integrate_dop853<1>(y, t, bp_ptr, num_bp, val0, val1, rtol, atol, initialStep, params, trackerMax, trackerMin, trajectory, integrationStatus);
    } else if (simMode == 2) {
        integrate_dop853<2>(y, t, bp_ptr, num_bp, val0, val1, rtol, atol, initialStep, params, trackerMax, trackerMin, trajectory, integrationStatus);
    }

    // Re-acquire the Python GIL after performing the integration and before before using the Python C-API
    py::gil_scoped_acquire acquire;

    // Package the results for Python
    std::vector<double> py_max_times, py_max_positions;
    std::vector<double> py_min_times, py_min_positions;
    if (max_peaks < 0) {
        // All valide peaks were tracked, already in cronological order
        py_max_times = max_t_mem;
        py_max_positions = max_z_mem;
        py_min_times = min_t_mem;
        py_min_positions = min_z_mem;
    } else {
        // Makes sure the fixed number of peaks tracked are in cronological order
        readPeaksTracker(trackerMax, py_max_times, py_max_positions);
        readPeaksTracker(trackerMin, py_min_times, py_min_positions);
    }

    // Return the Trajectory and Peaks as NumPy arrays and a tuple with Status
    return std::make_tuple(
        as_pyarray(std::move(trajectory.t)),
        as_pyarray(std::move(trajectory.z)),
        as_pyarray(std::move(trajectory.v)),
        as_pyarray(std::move(py_max_times)),
        as_pyarray(std::move(py_max_positions)),
        as_pyarray(std::move(py_min_times)),
        as_pyarray(std::move(py_min_positions)),
        std::make_tuple(trajectory.captured, integrationStatus.MODE, integrationStatus.nAccepted, integrationStatus.nRejected)
    );
}

// Create the Python module using pybind11
PYBIND11_MODULE(ode_cpp, m) {
    m.doc() = "C++ DOP853 Integrator for Acoustic Levitator ODEs with Peak Tracking";
    m.def("run_simulation", &run_simulation, 
          "Run the simulation of the acoustic levitator ODEs with peak tracking and return the trajectory and peaks.",
          py::call_guard<py::gil_scoped_release>(), // Allows the module to be executed in parallel along multiple threads during the integration
          py::arg("z0"), py::arg("v0"),
          py::arg("breakpoints"),
          py::arg("k"), py::arg("gEf"), py::arg("B"), py::arg("zEq"), py::arg("Lambda"),
          py::arg("val0") = 0.0, py::arg("val1") = 2.5, 
          py::arg("base_A") = 2.5, py::arg("base_phi") = 0.0, py::arg("dA") = 0.0, py::arg("omega") = 0.0,
          py::arg("max_peaks") = 100, py::arg("trackerStartTime") = 0.0, py::arg("simMode") = 0, 
          py::arg("rtol") = 1e-7, py::arg("atol") = 1e-10, py::arg("initialStep") = 1e-4);
}