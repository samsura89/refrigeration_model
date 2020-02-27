# Process to model the energy consumption of a fridge

# Strategy: 

Fridge's energy consumption is directly proportional to the time the compressor is on. Therefore, the aim is to model the length of the time the compressor is on.


Based on early impressions of the data (constrained by the time limit of 6 hours), it was decided to model the length of the time the compressor is on, based on 
1. the difference in temperatures inside the fridge when the door is opened and the temperature in the room (the latter was found to be the same as the user-defined setpoint, in most cases)
2. the length of time the door was open
This was unique for each device, but there could be a better fit if all of the data from each device were concatenated together

This model would be coupled by another model which predicts the door open length at a given hour of the day. This model was a probability distribution for the length of the door window at different times of the day - lunch time, dinner time, other time - for each device. This would be multiplied by a corresponding weight for the hour (for eg., the probability of door opening, for any length of time, at 1am is significantly lower than at 8pm or 2pm).


Once these model were developed, use the May 2019 hourly data of temperatures at LGA (https://www.ncdc.noaa.gov/cdo-web/datasets#NORMAL_HLY) in New York (it is assumed that the fridge is in New York, and that room temperature is equal to the outside temperature). 

The ipython notebooks in this repo show illustrate the the exploration and modelling, and the python scripts are used to automate the model building, and the actual prediction, for a given uuid and setpoint.

# Usage
 download the repo and run:
 
        python predict --uuid <uuid number> and --setpoint <setpoint>

# Assumptions/ Decisions:
1. Model can have 2 coupled parts: predicting compressor being on durations, and and predicting door open lengths
2. Each device is owned by different individual, who have their own model of keeping the door open, thereby, 


# Shortcomings/ Scope for improvement:
1. Fully automate the training pipeline - I was only able to automate the compressor-turned-on durations given door opening lengths and temperature differences, while I individually trained the probability durations for each uuid's door open length

2. The model was overfitted for one of the uuid, and did not transfer well across the others. Predictions could not be obtained for one of the uuid. The energy predictions were off, because the doors could be modelled much much better, and the compressor model was overfitted (in the time constraints, I could manage to only analyze the uuid "09ac4a10-7e8e-40f3-a327-1f93a5cf2383")

3. Investigate LSTM for better time-series prediction 

4. Use facebook's prophet package for timeseries prediction

5. A few more hours would have ensured better results, because the whole pipeline is kind of in place, and refining the overall predictions would be easier. 

6. Saved the device specific information more efficiently.


