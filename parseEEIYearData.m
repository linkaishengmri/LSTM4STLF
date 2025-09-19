function [weeklyData, maxLoad] = parseEEIYearData(filename, year)
    % Read file content
    fid = fopen(filename, 'r');
    if fid == -1
        error('Cannot open file: %s', filename);
    end
    
    lines = {};
    while ~feof(fid)
        lines{end+1} = fgetl(fid);
    end
    fclose(fid);
    
    % Determine number of spaces after date identifier (using the first line)
    firstLine = lines{1};
    dataStart = find(~isspace(firstLine(10:end)), 1) + 9;
    if isempty(dataStart)
        error('Cannot determine data start position');
    end
    
    % Parse data
    nLines = length(lines);
    allLoads = [];
    allDates = [];
    currentDate = [];
    currentLoads = [];
    
    for i = 1:nLines
        line = lines{i};
        if length(line) < 9
            fprintf('Skip invalid line %d: %s\n', i, line);
            continue;
        end
        
        % Parse date part
        dateStr = line(1:9);
        mm = str2double(dateStr(1:2));
        dd = str2double(dateStr(3:4));
        yyyy = str2double(dateStr(5:8));
        block = str2double(dateStr(9));
        
        % Parse load data (skip spaces after date)
        dataPart = line(dataStart:end);
        
        % Handle possible spaces (for values < 10000)
        loadValues = [];
        for j = 1:12
            startPos = (j-1)*5 + 1;
            if startPos > length(dataPart)
                break;
            end
            endPos = min(startPos+4, length(dataPart));
            valStr = dataPart(startPos:endPos);
            
            % Remove spaces and convert to number
            valStr = strrep(valStr, ' ', '');
            if isempty(valStr)
                loadValue = NaN;
            else
                loadValue = str2double(valStr);
            end
            loadValues(end+1) = loadValue;
        end
        
        % Store parsed results
        if block == 1
            % First 12-hour block
            currentDate = datetime(yyyy, mm, dd);
            currentLoads = loadValues;
        else
            % Second 12-hour block, combine to form full day
            if ~isempty(currentLoads) && currentDate == datetime(yyyy, mm, dd)
                fullDayLoads = [currentLoads, loadValues];
                allLoads = [allLoads; fullDayLoads];
                allDates = [allDates; currentDate];
            else
                fprintf('Warning: Date block mismatch or missing first block, line %d\n', i);
            end
            currentLoads = [];
        end
    end
    
    % Organize data by week (starting from Monday)
    % Find the first Monday on/after Jan 1 of the given year
    firstDay = datetime(year, 1, 1);
    dayOfWeek = weekday(firstDay); % Sunday=1, Monday=2, ...
    daysToAdd = mod(2 - dayOfWeek, 7); % number of days to add to reach Monday (0..6)
    firstMonday = firstDay + days(daysToAdd);
    
    % Try to find the first Monday that actually exists in data
    % (data might start after Jan 1 or some days missing)
    mondayIdx = find(allDates == firstMonday, 1);
    if isempty(mondayIdx)
        % find the first Monday in the dataset that is >= firstMonday
        mondayIdx = find(weekday(allDates) == 2 & allDates >= firstMonday, 1);
        if isempty(mondayIdx)
            % as a last resort, find the first Monday in allDates (if data starts later)
            mondayIdx = find(weekday(allDates) == 2, 1);
            if isempty(mondayIdx)
                error('Cannot find any Monday in the data for year %d', year);
            else
                warning('First Monday for year %d not present; using first Monday found in data: %s', year, datestr(allDates(mondayIdx)));
            end
        end
    end
    
    % Calculate total weeks
    nDays = length(allDates);
    nWeeks = floor((nDays - mondayIdx + 1) / 7);
    
    % Extract data from first Monday
    dataFromFirstMonday = allLoads(mondayIdx:end, :);
    
    % Ensure we have enough data for complete weeks
    if size(dataFromFirstMonday, 1) < nWeeks * 7
        nWeeks = nWeeks - 1;
    end
    
    % Reshape data into weekly structure (nWeeks x 168)
    weeklyData = zeros(nWeeks, 168);
    for i = 1:nWeeks
        startIdx = (i-1)*7 + 1;
        endIdx = i*7;
        weekData = dataFromFirstMonday(startIdx:endIdx, :);
        weeklyData(i, :) = reshape(weekData', 1, 168);
    end
    
    % Find maximum load for normalization (omit NaNs)
    maxLoad = max(weeklyData(:), [], 'omitnan');
end
