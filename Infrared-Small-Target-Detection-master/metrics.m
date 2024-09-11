function [mIoU, POD, FAR] = evaluateSegmentation(predictedMask, actualMask)
    % Check if masks have the same size, and resize if necessary
    if ~isequal(size(predictedMask), size(actualMask))
        warning('Masks are of different sizes. Resizing predictedMask to match actualMask.');
        predictedMask = imresize(predictedMask, size(actualMask), 'nearest');
    end

    % Ensure the masks are logical
    predictedMask = logical(predictedMask);
    actualMask = logical(actualMask);

    % Calculate Intersection and Union
    intersection = predictedMask & actualMask;
    union = predictedMask | actualMask;
    
    % Calculate Mean Intersection over Union (mIoU)
    if sum(union(:)) == 0
        mIoU = 1; % If both masks are empty, IoU is considered perfect (1.0)
    else
        mIoU = sum(intersection(:)) / sum(union(:));
    end

    % Calculate Probability of Detection (POD)
    if sum(actualMask(:)) == 0
        POD = 1; % If actual mask is empty, POD is considered perfect (1.0)
    else
        POD = sum(intersection(:)) / sum(actualMask(:));
    end

    % Calculate False Alarm Rate (FAR)
    if sum(predictedMask(:)) == 0
        FAR = 0; % If no predictions are made, FAR is zero
    else
        falseAlarm = predictedMask & ~actualMask;
        FAR = sum(falseAlarm(:)) / sum(predictedMask(:));
    end
end

path1 = "C:/Users/chand/Desktop/Project 2/DNA-Net/Infrared-Small-Target-Detection-master/img_demo/Misc_2_Pred.png";
path2 = "C:/Users/chand/Desktop/Project 2/U-Net/Pytorch-UNet-master/data/masks/Misc_2_pixels0.png";
predictedMask = imread(path1) ;
actualMask = imread(path2);

% Evaluate the segmentation
[mIoU, POD, FAR] = evaluateSegmentation(predictedMask, actualMask);

% Display results
fprintf('Mean IoU: %.2f\n', mIoU);
fprintf('Probability of Detection: %.2f\n', POD);
fprintf('False Alarm Rate: %.2f\n', FAR);