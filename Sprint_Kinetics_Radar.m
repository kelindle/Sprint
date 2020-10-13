clc; clear;

%% Data Import
directory = uigetdir; %Choose directory with radar data files 
files = dir(directory); %Choose spreadsheet with supplemental height and weight information 

%Get file with height and weight data for all files in directory 
heightWeightFile = uigetfile('*.xlsx');
heightWeightData = readtable(heightWeightFile);

WRmass = 0; %Change if wearable resistance is implemented on the recorded trials 

heights = heightWeightData.Height; %Extracting heights from supplementary .xls
weights = heightWeightData.Weight; %Extracting weights from supplementary .xls

%Initializing arrays to store kinetic values for all of the trials 
heights_output = zeros(length(files)-2,1); 
weights_output = zeros(length(files)-2,1); 
Fnaughts = zeros(length(files)-2,1); 
relativeFnaughts = zeros(length(files)-2,1);
maxVelos = zeros(length(files)-2,1);
experimentalMaxVelos = zeros(length(files)-2,1);
maxPowers = zeros(length(files)-2,1);
experimentalMaxAccelerations = zeros(length(files)-2,1);
theoreticalMaxAccelerations = zeros(length(files)-2,1);
relativemaxPowers = zeros(length(files)-2,1);
FVslopes = zeros(length(files)-2,1);
relativeFVslopes = zeros(length(files)-2,1);
RFDecreaserates = zeros(length(files)-2,1);
meter5times = zeros(length(files)-2,1);
meter10times = zeros(length(files)-2,1);
meter20times = zeros(length(files)-2,1);
meter30times = zeros(length(files)-2,1);
Taus = zeros(length(files)-2,1);
Vhmaxs = zeros(length(files)-2,1);
veloDataLengths = zeros(length(files)-2,1);
correlationCoefficients = zeros(length(files)-2,1);
vhExperiments = zeros(length(files)-2,100);
vhTheories = zeros(length(files)-2,100);
rfMaxes = zeros(length(files)-2,1);
vnaughts = zeros(length(files)-2,1);
trimmed_samples = zeros(length(files)-2,1);
trimmed_position = zeros(length(files)-2,1);


for i = 3:length(files)

file = files(i).name; %Get current file
    
%Load data files Radar
data = readtable(file); %Read whole data file
data = rmmissing(data); %Remove emply first column of data
% samplerate = 46.875; %Set sample rate, provided in the radar documentation
samplerate = 333; %Set sample rate, provided in the radar documentation
%velocitydata = transpose(data.Speed); %Save speed data column
% velocitydata = transpose(data.Speed);%*0.44704; %converting from mph to mps
velocitydata = transpose(data.Velo);%*0.44704; %converting from mph to mps
exp_t = transpose(data.Time); %Initialize time array

%Load data files 1080
% data = readtable(file); %Read whole data file
% samplerate = 333; %Set sample rate, provided in the radar documentation
% velocitydata = transpose(data.Speed_m_s_); %Read velocity data
% sample_duration = transpose(data.SampleDuration_s_); %Read sample duration
% exp_t = cumtrapz(1:length(sample_duration), sample_duration); %Create time array with sample durations

plot(data.Velo); %plot for user input

user_input_min = ginput(1); %get user input for minimum before start of sprint
user_input_min = user_input_min(1); %choose x value from user input

if user_input_min == 0
    user_input_min = 1;
end

user_trimmed_velo = data.Velo(round(user_input_min,0):end); %trim velo based on user input
user_trimmed_time = data.Time(round(user_input_min,0):end); %trim time based on user input

smoothed_velo = smooth(user_trimmed_velo, 5);

%Finding the Starting Sample if data is untrimmed
%startingsample = 1; %Use the first sample, radar data is trimmed in the stalker software
% startingsample = find(smoothed_velo>1.5,1);

threshold = 0.5; % m/s
startingsample = 1;
for p = find(smoothed_velo == max(smoothed_velo), 1):-1:2
    if smoothed_velo(p) > threshold && smoothed_velo(p-1) < threshold
        startingsample = p;
        break
    end
end

temp_trimmed_velo = user_trimmed_velo(startingsample:end); %trim user trimmed velo data with starting point found
temp_trimmed_time = user_trimmed_time(startingsample:end); %trim user trimmed time data with starting point found

calcposition = cumtrapz(temp_trimmed_time,temp_trimmed_velo); %Calculate position array by integrating speed data, used for trimming data

triallength = 30; %30m sprints
endingsample = find(calcposition>triallength,1); %Trim the end of the data to 1 sample after the participant reaches 30 m

if endingsample > length(user_trimmed_velo)
    endingsample = length(user_trimmed_velo);
end

if isempty(endingsample)
    endingsample = length(calcposition); %If recording is ended before the participant reaches 30 m, do not trim the data
end

vhexperiment = user_trimmed_velo(startingsample:endingsample);
texperiment = user_trimmed_time(startingsample:endingsample);
t_fit = texperiment - texperiment(1);

plot(vhexperiment) %going to comment this out, could just be a good checker upper

fit = genfit(t_fit,vhexperiment,'accelerationfit_c',[max(vhexperiment) 1.2 1.5],'nonlin', [1 0 1]); %fit with a y offset
exp_time = data.Time - user_trimmed_time(startingsample); %offset data to starting sample
v_theory_initial = fit.solution(1)*(1-exp(-exp_time/fit.solution(2)))+fit.solution(3); %create full model though we will cut it on the plots when it crosses zero

global_starting_sample = round(user_input_min,0)+startingsample;
global_ending_sample = round(user_input_min,0)+startingsample+endingsample;

if global_ending_sample > length(data.Velo)
    global_ending_sample = length(data.Velo);
end

plot_vhexperiment_initial = data.Velo(global_starting_sample:global_ending_sample);
plot_texperiment_initial = data.Time(global_starting_sample:global_ending_sample);
v_theory_initial = v_theory_initial(1:global_ending_sample);

vhtheory_origin_initial = find(v_theory_initial > 0,1);
plot_exp_time = exp_time + plot_texperiment_initial(1);
plot_exp_time = plot_exp_time(1:global_ending_sample);

greater_than_zero = find(v_theory_initial > 0);
trimmed_points = length(greater_than_zero) - length(plot_vhexperiment_initial);

estimated_trimmed_position = cumtrapz(plot_exp_time(greater_than_zero(1):greater_than_zero(1)+11), v_theory_initial(greater_than_zero(1):greater_than_zero(1)+11));

%plot to check fit

plot(plot_texperiment_initial, plot_vhexperiment_initial) %dialed
hold on
plot(plot_exp_time, v_theory_initial)
% plot(user_trimmed_time, smoothed_velo) %Uncomment this if you wanna see the
% smoothed velo
title(strcat('Trimmed Radar Data vs. Model '))
xlabel('Time (s)')
ylabel('Velocity (m/s)')
xlim([0 10])
ylim([0 10])
hold off
pause(3)
close all

Vhmax_initial = fit.solution(1)+fit.solution(3);
Tau_initial = fit.solution(2);

v_theory_trimmed = v_theory_initial(vhtheory_origin_initial:end);
t_final = (1:length(v_theory_trimmed))/samplerate;
vhexperiment_initial = plot_vhexperiment_initial;

offset = length(v_theory_trimmed) - length(vhexperiment_initial);

initial_residuals = vhexperiment_initial - v_theory_trimmed(end-length(vhexperiment_initial)+1:end);
vhtheory_outlier_fixer = v_theory_initial(end-length(vhexperiment_initial)+1:end);

mean_residual = mean(initial_residuals);
sd_residual = std(initial_residuals);

outliers = [];

for k = 1:length(initial_residuals)
    if initial_residuals(k) > 2*sd_residual | initial_residuals(k) < -1*2*sd_residual %outside 
        outliers = [outliers k];
    end
end

vhexperiment = vhexperiment_initial;
exp_t_final = (1:length(vhexperiment))/samplerate + offset/samplerate;

if isempty(outliers)
    Vhmax = Vhmax_initial;
    Tau = Tau_initial;
else
    for j = 1:length(outliers)
        
%         vhexperiment(outliers(j)) = vhtheory_outlier_fixer(outliers(j));
        vhexperiment(outliers(j)) = [];
        exp_t_final(outliers(j)) = [];
        outliers = outliers - 1; %Decrement outlier location because of removed points
        
    end
    
%     t_fit_2 = transpose((1:length(vhexperiment))/samplerate);
%     fit = genfit(t_fit_2,vhexperiment,'accelerationfit_c',[max(vhexperiment) 1.2 1.5],'nonlin', [1 0 1]); %Fit raw speed data to the exponential model
    fit = genfit(exp_t_final,vhexperiment,'accelerationfit_c',[max(vhexperiment) 1.2 1.5],'nonlin', [1 0 1]); %Fit raw speed data to the exponential model
    vhtheory = fit.solution(1)*(1-exp(-t_final/fit.solution(2))) + fit.solution(3); %Initialize theoretical velocity using the optimal model parameters
    Vhmax = fit.solution(1)+fit.solution(3); %Extract max velocity parameter from model
    Tau = fit.solution(2); %Extract time constant parmaeter from model
end

vhtheory_origin = find(vhtheory > 0, 1);
vhtheory = vhtheory(vhtheory_origin:end);
t_final = t_final(vhtheory_origin:end);

plot(t_final, vhtheory)
hold on
plot(exp_t_final, vhexperiment)
title(file)
hold off
pause(2)
close all

% disp(Vhmax_initial-Vhmax);
% disp(Tau_initial-Tau);

%Finally calculate the other associated signals
ahtheory = diff(vhtheory)/(1/samplerate); %Calculate theoretical acceleration by differentiating theoretical velocity
ahexperiment = diff(vhexperiment)/(1/samplerate); %Calculate experimental velcoity by differentiating trimmed experimental speed data
phtheory = cumtrapz(t_final,vhtheory); %Calculate theoretical position data by integrating theoretical velocity data; used for figures

%% Force Calculations

%Constants
h = heights(i-2)/100; %Enter in cm, convert to meters
m = weights(i-2)*.454+WRmass; %Enter weight in lbs, mass of WR in kg. Convert lbs to kg
T = 20.0; %Enter in C, Temperature inside of the field house during data collection
Pb = 760; %Enter in Torr, Default
vw = 0; %wind speed in m/s, Data collection was done indoors
Cd = 0.9; %Drag Coefficient, constant

%F-V Calculations
Fvert = zeros(1,length(t_final)) + m*9.81; %vertical force estimated as participants body weight 
ah = (Vhmax./Tau).*exp(-t_final./Tau); %Horizontal acceleration using optimal model parameters
Af = (.2025*h.^.725 * (m.^.425))*.266; %Estimated frontal surface area of the patient using height and body mass
rho = 1.293*(Pb/760)*(273/(273+T)); %Air density 
k = .5*rho*Af*Cd; %Air resistance coefficient
Faero = k.*((vhtheory-vw).^2); %Air resistance force 
Fh = m.*ah+Faero; %Calculated horizontal force using theoretical values from model
HorizontalForceSystemNormalized = Fh/m; %Normalizing horizontal force by total mass (body mass plus any wearable resistance)
fvnfit = polyfit(vhtheory, HorizontalForceSystemNormalized, 1); %Linear fit for estimating normalized force-velcoity profile
fvfit = polyfit(vhtheory, Fh, 1); %Linear fit for estimating force-velocity profile
P = Fh .* vhtheory; %Calculating theoretical power using horizontal force * horizontal velocity 
PSystemNormalized = HorizontalForceSystemNormalized .* vhtheory; %Calculating theoretical normalized power using normalized horizontal force * horizontal force
RF = (Fh(round(.3*samplerate,0):end)./sqrt(Fh(round(.3*samplerate,0):end).^2 + (m*9.81).^2))*100; %Calculating ration of horizontal force to total force
Drffit = polyfit(vhtheory(round(.3*samplerate,0):end),(Fh(round(.3*samplerate,0):end)./sqrt(Fh(round(.3*samplerate,0):end).^2+Fvert(round(.3*samplerate,0):end).^2))*100,1); %Linear fit of RF as a function of a velocity after .3 seconds as described by Samozino et al. 2016
Drf = Drffit(1); %Slope of RF linear fit as a function of velocity 
Sfv = fvfit(1); %Slope of absolute force-velocity profile
rSfvSys = fvnfit(1);%Slope of normalized force-velocity profile
F0 = fvfit(2); %Theoretical maximum force produced in the horizontal direction; first value in the horizontal force array, as it is a decreasing signal
V0 = (0-fvfit(2))/fvfit(1);
Vmax = max(vhtheory); %Maximum velocity in the theoretical velocity array
Pmax = max(P); %Maximum value in the power signal 
rPmaxSys = max(PSystemNormalized); %Maximum value in the normalize power signal 

%Split times
%Indexes of passing 5, 10, 20, and 30 meter marks
% index5 = find(round(calcposition,0)==5); index10 = find(round(calcposition,0)==10); index20 = find(round(calcposition,0)==20); index30 = find(round(calcposition,0)==30); %Indeces of 10,20, and 30m splits using integrated experimental velocity 
index5 = find(round(phtheory,0)==5); index10 = find(round(phtheory,0)==10); index20 = find(round(phtheory,0)==20); index30 = find(round(phtheory,0)==30); %Indeces of 10,20, and 30m splits using integrated experimental velocity 

%Time values of passing 5, 10, 20, and 30 meter marks
% meter5 = exp_t(index5(1)); meter10 = exp_t(index10(1)); meter20 = exp_t(index20(1)); meter30 = exp_t(index30(1));%Time at each of the splits 
meter5 = t_final(index5(1)); meter10 = t_final(index10(1)); meter20 = t_final(index20(1)); 

if isempty(index30)
    meter30 = 0;
else
    meter30 = t_final(index30(1));%Time at each of the splits 
end

%Open subplot at maximum window size, save, close 
%All plots
fig = figure('units','normalized','outerposition',[0 0 1 1]); %Open figure at maximize window size 
distance = cumtrapz(exp_t_final, vhexperiment);
subplot(2,3,1)
plot(t_final, phtheory,'k','LineWidth',2)
hold on
plot(exp_t_final,distance);
hold off
title('Position');
xlabel('Time (s)');
ylabel('Position (m)');
subplot(2,3,2)
plot(t_final,vhtheory,'k','LineWidth',2);
hold on
plot(exp_t_final,vhexperiment);
hold off
title('Velocity')
ylabel('Sprint Velocity (m/s)')
xlabel('Time (s)')
ylim([0 10]);
subplot(2,3,3)
plot(exp_t_final(1:end-1),ahexperiment);
hold on ;
plot(t_final(1:end-1),ahtheory,'k','LineWidth',2);
hold off;
title('Acceleration');
xlabel('Time (s)');
ylabel('Acceleration (m/s/s)');
ylim([-1 10]);
subplot(2,3,4)
plot(vhtheory, Fh);
title('Force-Velocity Profile');
xlabel('Velocity (m/s)');
ylabel('Horizontal Force (N)');
subplot(2,3,5)
hold on;
yyaxis left;
plot(vhtheory, HorizontalForceSystemNormalized);
title('Relative Force and Power');
xlabel('Velocity (m/s)');
ylabel('Relative Horizontal Force (N/kg)');
yyaxis right;
plot(vhtheory, PSystemNormalized);
ylabel('Relative Power (W/kg)');
hold off;
subplot(2,3,6)
plot(vhtheory(round(.3*samplerate,0):end),RF)
title('Ratio of Horizontal Force');
xlabel('Sprint Velocity (m/s)');
ylabel('Ratio of Horizontal Force (%)');
pause(3)

%Save and close figure
saveas(fig,[files(i).name(1:end-4) '.png']); %Saving plots as pngs for future reference 
close(fig); %Close figure 

%Correlation coefficients between experimental and theoretical velocity and
%position

correlationVelo = corrcoef(vhexperiment, vhtheory(length(vhtheory)-length(vhexperiment)+1:length(vhtheory)));
correlationPosition = corrcoef(distance, phtheory(length(phtheory)-length(distance)+1:length(phtheory)));


%Save workspace variables
%save([file(1:end-4) '.mat']); 

%Saving current instance of all kinetics and split times 
Fnaughts(i-2) = F0;
vnaughts(i-2) = V0;
relativeFnaughts(i-2) = fvnfit(2);
maxVelos(i-2) = Vmax;
maxPowers(i-2) = Pmax;
relativemaxPowers(i-2) = rPmaxSys;
FVslopes(i-2) = Sfv;
relativeFVslopes(i-2) = rSfvSys;
RFDecreaserates(i-2) = Drf;
meter5times(i-2) = meter5; meter10times(i-2) = meter10; meter20times(i-2) = meter20; meter30times(i-2) = meter30;
Taus(i-2) = Tau;
Vhmaxs(i-2) = Vhmax;
veloDataLengths(i-2) = length(vhexperiment);
correlationCoefficients(i-2) = correlationVelo(1,2);
rfMaxes(i-2) = max(RF);
experimentalMaxVelos(i-2) = max(vhexperiment);
experimentalMaxAccelerations(i-2) = max(ahexperiment);
theoreticalMaxAccelerations(i-2) = max(ahtheory);
heights_output(i-2) = h;
weights_output(i-2) = m;
trimmed_samples(i-2) = trimmed_points;
trimmed_position(i-2) = estimated_trimmed_position(end);

%Interpolate velocity data to 100 nodes for comparison with radar data
registeredExpTime = linspace(1,length(exp_t_final),100);
registeredTheoryTime = linspace(1,length(t_final),100);
vhExperiments(i-2,:) = interp1(vhexperiment,registeredExpTime);
vhTheories(i-2,:) = interp1(vhtheory,registeredTheoryTime);

end 

%Initialize struct for compilation of kinetics and split times 
FinalData = struct('Names',files(3).name(1:end-4),'Height',heights_output(1),'F0',Fnaughts(1),'rF0',relativeFnaughts(1),'Weight',weights_output(1),'V0', vnaughts(1),'Vmax',maxVelos(1),...
    'Pmax',maxPowers(1),'rPmax',relativemaxPowers(1),'Sfv',FVslopes(1),'rSfv',relativeFVslopes(1),'Drf',...
    RFDecreaserates(1),'Correlation',correlationCoefficients(1),'Split_5m',meter5times(1),'Split_10m',...
    meter10times(1),'Split_20m',meter20times(1),'Split_30m',meter30times(1),'Exp_Max_Velos', experimentalMaxVelos(1),...
    'Exp_Max_Accel', experimentalMaxAccelerations(1), 'Theory_Max_Accel', theoreticalMaxAccelerations(1), 'Tau', Taus(1),...
    'Vhmax', Vhmaxs(1),'RF_Max', rfMaxes(1), 'Trimmed_Samples', trimmed_samples(1), 'Trimmed_Positions', trimmed_position(1));

for i = 2:length(files)-2
    
    %Append the rest of the trials on to the initialized struct 
    NextData = struct('Names',files(i+2).name(1:end-4),'Height',heights_output(i),'F0',Fnaughts(i),'rF0',relativeFnaughts(i),'Weight',weights_output(i),'V0',vnaughts(i),'Vmax',maxVelos(i),...
    'Pmax',maxPowers(i),'rPmax',relativemaxPowers(i),'Sfv',FVslopes(i),'rSfv',relativeFVslopes(i),'Drf',...
    RFDecreaserates(i),'Correlation',correlationCoefficients(i),'Split_5m',meter5times(i),'Split_10m',...
    meter10times(i),'Split_20m',meter20times(i),'Split_30m',meter30times(i),'Exp_Max_Velos', experimentalMaxVelos(i),...
    'Exp_Max_Accel', experimentalMaxAccelerations(i), 'Theory_Max_Accel', theoreticalMaxAccelerations(i), 'Tau', Taus(i),...
    'Vhmax', Vhmaxs(i),'RF_Max', rfMaxes(i), 'Trimmed_Samples', trimmed_samples(i), 'Trimmed_Positions', trimmed_position(i));

    FinalData = [FinalData, NextData];
    
end

%save('Velocities Radar', 'vhTheories', 'vhExperiments');

%Save and export struct as a .csv
writetable(struct2table(FinalData),'Radar Data.csv');
