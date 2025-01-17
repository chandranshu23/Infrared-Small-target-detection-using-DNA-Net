% Load the input image and mask
infraredImage = imread('C:/Users/chand/Desktop/Project 2/DNA-Net/Infrared-Small-Target-Detection-master/img_demo/Misc_90.png');
labelImage = imread('C:/Users/chand/Desktop/Project 2/DNA-Net/Infrared-Small-Target-Detection-master/img_demo/Misc_90_Pred.png'); % Assuming the mask is binary with bounding boxes

% Ensure both images are in grayscale
if size(infraredImage, 3) > 1
    infraredImage = rgb2gray(infraredImage);
end
if size(labelImage, 3) > 1
    labelImage = rgb2gray(labelImage);
end

% Resize the label image to match the dimensions of the infrared image
resizedLabel = imresize(labelImage, [size(infraredImage, 1), size(infraredImage, 2)]);

% Convert the label image to binary
binaryLabel = imbinarize(resizedLabel);

% Find the connected components in the binary label image
connectedComponents = bwconncomp(binaryLabel);

% Get the properties of each connected component (target)
stats = regionprops(connectedComponents, 'BoundingBox', 'Centroid');

% Initialize the output image
outputImage = cat(3, infraredImage, infraredImage, infraredImage); % Convert to RGB for colored bounding boxes

% Define the padding value (e.g., 5 pixels)
padding = 5;

% Loop through each target and draw a bounding box around it with padding
for i = 1:length(stats)
    bbox = stats(i).BoundingBox;
    
    % Add padding to the bounding box
    xMin = max(bbox(1) - padding, 1);
    yMin = max(bbox(2) - padding, 1);
    width = min(bbox(3) + 2 * padding, size(infraredImage, 2) - xMin);
    height = min(bbox(4) + 2 * padding, size(infraredImage, 1) - yMin);
    
    paddedBbox = [xMin, yMin, width, height];
    
    % Draw the padded bounding box on the output image
    outputImage = insertShape(outputImage, 'Rectangle', paddedBbox, 'Color', 'red', 'LineWidth', 2);
    
    % Optionally, mark the centroid of the target
    centroid = stats(i).Centroid;
    outputImage = insertMarker(outputImage, centroid, 'o', 'Color', 'red', 'Size', 5);
end

% Display the final image with padded bounding boxes and centroids (if added)
imshow(outputImage);
title('Infrared Image with Padded Bounding Boxes Around Labeled Targets');
output_path = "C:/Users/chand/Desktop/Project 2/DNA-Net/Infrared-Small-Target-Detection-master/img_demo/Misc_90_Pred_boxes.png";
imwrite(outputImage, output_path);