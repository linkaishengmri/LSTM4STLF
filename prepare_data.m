%% EEI Load Data Processing for LSTM Model (fixed split & shapes)
% Process multi-year EEI load data into train/test sets for weekly LSTM

%% Init
clear; clc; close all;

%% Load yearly data
years = 2016:2024;
allWeeklyData = [];   % weekly data across years
allMaxLoads = [];     % yearly max loads
yearWeeks = zeros(size(years)); % weeks per year

for idx = 1:numel(years)
    y = years(idx);
    filename = sprintf('%d_eei_loads.txt', y);
    fprintf('Processing %s...\n', filename);
    
    if ~exist(filename, 'file')
        fprintf('Missing %s, skip year %d\n', filename, y);
        continue;
    end
    
    try
        [weeklyData, maxLoad] = parseEEIYearData(filename, y);
        nW = size(weeklyData, 1);
        yearWeeks(idx) = nW;
        allWeeklyData = [allWeeklyData; weeklyData];
        allMaxLoads = [allMaxLoads; maxLoad];
        fprintf('Processed %d weeks for %d\n', nW, y);
    catch ME
        fprintf('Error %s: %s\n', filename, ME.message);
    end
end

% Check data
if isempty(allWeeklyData)
    error('No weekly data loaded.');
end

% Normalize by global max
globalMaxLoad = max(allMaxLoads);
if isempty(globalMaxLoad) || isnan(globalMaxLoad) || globalMaxLoad == 0
    error('Invalid global max load.');
end
allWeeklyDataNormalized = allWeeklyData / globalMaxLoad;

fprintf('Global max load: %.2f\n', globalMaxLoad);
fprintf('Total weeks: %d (sum=%d)\n', size(allWeeklyDataNormalized,1), sum(yearWeeks));

%% Build LSTM sequences
% Input: 52 weeks (168x52), Output: 1 week (168)

trainYears = 2016:2022;
testYears  = 2023:2024;

% Find train/test year indices
[tf_train, ~] = ismember(trainYears, years);
if ~all(tf_train)
    warning('Some train years missing.');
end
trainIdxs = find(ismember(years, trainYears));
trainEndIdx = sum(yearWeeks(trainIdxs));
testIdxs = find(ismember(years, testYears));
testStartIdx = trainEndIdx + 1;

fprintf('trainEndIdx = %d\n', trainEndIdx);
fprintf('testStartIdx = %d\n', testStartIdx);

% Seq arrays
X = [];        % 168x52xN
X_month = [];  % 4x52xN
Y = [];        % Nx168

totalWeeks = size(allWeeklyDataNormalized, 1);

% Need 52 weeks history
firstTargetWeek = 53;
if totalWeeks < firstTargetWeek
    error('Not enough weeks (%d).', totalWeeks);
end

% Assign months (4-week blocks)
allMonthAssignments = zeros(1, totalWeeks);
for i = 1:totalWeeks
    block = ceil(i / 4);
    month = mod(block - 1, 12) + 1;
    allMonthAssignments(i) = month;
end

for i = firstTargetWeek:totalWeeks
    % Input sequence
    inputSeq = allWeeklyDataNormalized(i-52:i-1, :);
    X = cat(3, X, inputSeq');  
    
    % Month features
    monthFeatures = zeros(4, 52);
    for w = 1:52
        weekIdx = i - 52 + w - 1;
        grayCode = monthToGrayCode(allMonthAssignments(weekIdx));
        monthFeatures(:, w) = grayCode';
    end
    X_month = cat(3, X_month, monthFeatures);
    
    % Output week
    Y = [Y; allWeeklyDataNormalized(i, :)];
end

nSamples = size(X, 3);
fprintf('Created %d samples\n', nSamples);
fprintf('X shape: %d x %d x %d\n', size(X,1), size(X,2), size(X,3));
fprintf('Month shape: %d x %d x %d\n', size(X_month,1), size(X_month,2), size(X_month,3));
fprintf('Y shape: %d x %d\n', size(Y,1), size(Y,2));

%% Split train/test
testStartSeqIdx = testStartIdx - 52;
if testStartSeqIdx < 1
    error('Bad test start idx (%d).', testStartSeqIdx);
end
if testStartSeqIdx > nSamples
    error('Too few samples (%d).', nSamples);
end

if testStartSeqIdx > 1
    X_train = X(:,:,1:testStartSeqIdx-1);
    X_train_month = X_month(:,:,1:testStartSeqIdx-1);
    Y_train = Y(1:testStartSeqIdx-1, :);
else
    X_train = zeros(size(X,1), size(X,2), 0);
    X_train_month = zeros(size(X_month,1), size(X_month,2), 0);
    Y_train = zeros(0, size(Y,2));
end

X_test = X(:,:,testStartSeqIdx:end);
X_test_month = X_month(:,:,testStartSeqIdx:end);
Y_test = Y(testStartSeqIdx:end, :);

fprintf('Train: %d\n', size(X_train,3));
fprintf('Test: %d\n', size(X_test,3));

%% Save
save('load_dataset.mat', 'X_train','X_train_month','Y_train','X_test','X_test_month','Y_test','globalMaxLoad','-v7.3');
fprintf('Saved load_dataset.mat\n');

%% Visualize samples
if size(X_train,3) > 0
    figure('Position', [100, 100, 1200, 800]);
    nShow = min(4, size(X_train,3));
    for k = 1:nShow
        subplot(2,2,k);
        seqIdx = randi(size(X_train, 3));
        past52 = squeeze(X_train(:,:,seqIdx));
        plot(1:168, past52(:,1:52)); 
        hold on;
        plot(1:168, Y_train(seqIdx, :), 'r', 'LineWidth', 2);
        hold off;
        title(sprintf('Sample %d', k));
        xlabel('Hour (1..168)');
        ylabel('Norm Load');
        grid on;
    end
else
    warning('No train samples to plot.');
end

%% Month Gray code map
fprintf('\nMonth to Gray:\n');
for m = 1:12
    gc = monthToGrayCode(m);
    fprintf('%d\t%s\n', m, num2str(gc));
end

fprintf('\nFirst 52 week month assignment:\n');
for w = 1:52
    fprintf('%d\t%d\n', w, allMonthAssignments(w));
end
