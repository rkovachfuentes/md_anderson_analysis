by @rkovachfuentes
Last modified 09/29/26

Start by using "general_analysis.py". Most of the other files are variations of these that were specialized to a given dataset.

config.py controls parameters like debugging messages (verbose), booleans to show or hide plots, etc.

general_analysis.py has the following structure:
generator
    denoise_and_get_area
        get_drop_event
        calculate_drop_area
        calculate_drop_area_sensitive
        convert_dose

along with some optional helper functions like inspect_point for later analysis.

denoise_and_get_area, using its helper functions, takes in an input csv file containing one waveform. get_drop_event identifies first cleans the signal from extraneous spikes that are not part of the signal (if present). It then dynamically calculates a threshold for statistically significant signal (by default, 4 times the noise range in the first 500 data points; this can be adjusted by the user, if desired). If no signal is identified, the function returns that no signal was detected. If a signal is found, the function returns the location where the signal jumps sharply away from baseline. calculate_drop_area and calculate_drop_area_sensitive each calculate the signal end and total area, saving each waveform as an image for visual inspection of the area and threshold, if desired. See the sample images for examples.

generator is a wrapper which calls denoise_and_get_area for every relevant csv file contained within a larger log file for a given experiment. Log files usually contain hundreds or thousands of waveforms. "Relevant" csv files are selected using HV and sensor/detector filters. Look at the end of general_analysis.py to see an example of how the generator function is called. Note that ALL file paths must be for the same date and formatted in the same way, otherwise the generator will not work.

To run, you will have to download the data and modify the expected filepaths:
data_path = path to data directory, containing a folder with the date name
log_file = exact path to log file for that dataset
output_file = desired output csv filename and path
outpath = desired directory to save output images to

The output will be a single csv file, with each row containing a single waveform and its corresponding area, dose, and other information contained in the original log file. This csv can then be plotted for the desired application.

Several functions have tunable parameters that may need to be adjusted depending on the signal characteristics. Finding the optimal values may take some trial and error. A few are noted here:
- parity: "positive" or "negative", tells the algorithm whether to look for a positive or negative signal. If this is set backwards you may get no signals detected!
- extrapolate_dose: True or False, flag with the option to extrapolate dose using distance data with an inverse square law fit to a provided dose csv file, if applicable
- sensitive_limit and baseline_noise are both parameters used to determine the dynamic threshold calculation and can be adjusted by the user, if desired
- there are some hardcoded values (e.g. max pulse value allowed of 10e-6) which can be adjusted or removed by the user. These values are to prevent unphysically long or short pulses from being detected.