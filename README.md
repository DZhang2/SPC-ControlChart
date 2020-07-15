# SPC-ControlChart
Python module for creating a variety of control charts

Charts included

* R and S charts for variability

* xBar chart for central tendency

* p and np charts for fraction non-conforming

* c and u charts for # of nonconformities

All charts include phase I and II capabilities for plotting and summary analysis for phase I

Iterative approach for handling OC samples during phase I by removing OC samples and recalculating control limits recursively until all samples are IC

Basic Methods for control charts

* summary()
* show_charts()
* show_phase2_chart()
* add_phase2_samples(samples)
* get_num_iterations()
* get_oc_samples()
* get_final_control_limits()

Initialization
* required parameter (samples)
* optional parameters: L (default to 3), title
