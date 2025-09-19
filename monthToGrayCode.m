function grayCode = monthToGrayCode(month)
    % Convert month (1-12) to 4-bit Gray code
    % Standard binary to Gray code conversion: G = B XOR (B >> 1)
    if month < 1 || month > 12
        error('Month must be between 1 and 12');
    end
    
    % Convert month to 4-bit binary (0-based index)
    binMonth = dec2bin(month-1, 4) - '0'; % 0-based indexing
    
    % Convert to Gray code
    grayCode = zeros(1, 4);
    grayCode(1) = binMonth(1);
    for i = 2:4
        grayCode(i) = xor(binMonth(i-1), binMonth(i));
    end
end