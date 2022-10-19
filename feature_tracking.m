%% Tracking of rocket using feature extraction
clear
close all

% Output video player
videoOut = vision.VideoPlayer('Name', 'Rocket Tracking', ...
    'Position', [640 480 640 480]);

% Read Video and make reference image
videoReader = vision.VideoFileReader('am_rocket1.mov');
i = 0;
% default 190th frame
while i < 190
    img_org = step(videoReader);
    i = i + 1;
end
% Convert to grayscale
img_gray = rgb2gray(img_org);
figure,
imshow(img_gray)

rocketPolygon = [1, 1;size(img_gray, 2), 1;...
    size(img_gray, 2), size(img_gray, 1);...
    1, size(img_gray, 1);1, 1];


% Detect Feature Points of reference
rocketPoints = detectSURFFeatures(img_gray);
% Show feature points
figure;
imshow(img_gray);
title('10 Feature Points from Rocket Image');
hold on;
plot(selectStrongest(rocketPoints, 10));

frame = step(videoReader);
frameSize = size(frame);
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);
n = 1;
flag = 1;
bbox = 0;
% Loop through image
while(n < 800)
    flag = 1;
    sceneframe = step(videoReader);
    sceneframe = rgb2gray(sceneframe);
    scenePoints = detectSURFFeatures(sceneframe);
    
    % Extract feature descriptors
    [refFeatures, rocketPoints] = extractFeatures(img_gray, rocketPoints);
    [sceneFeatures, scenePoints] = extractFeatures(sceneframe, scenePoints);
    
    % Match features using descriptors.
    rocketPairs = matchFeatures(refFeatures, sceneFeatures);
    
    if ~isempty(rocketPairs)
        matchedRocketPoints = rocketPoints(rocketPairs(:, 1), :);
        matchedScenePoints = scenePoints(rocketPairs(:, 2), :);
        
        try
            % Affine transformation
            [tform, inlierRocketPoints, inlierScenePoints] = ...
                estimateGeometricTransform(matchedRocketPoints, matchedScenePoints, 'affine');
        catch ME
            flag = 0;
        end
        
        % Transform polygon into coordinate system of the scene image.
        if flag
            newRocketPolygon = transformPointsForward(tform, rocketPolygon);
            
            % Display a bounding box around the detected face.
            xyPoints = [round(inlierScenePoints.Location) 20.*ones([length(round(inlierScenePoints.Location)) 1])];
            sceneframe = insertShape(sceneframe, 'Circle', xyPoints, 'LineWidth', 2, 'Color', 'green');
            % Display detected corners.
            sceneframe = insertMarker(sceneframe, round(inlierScenePoints.Location), '+', 'Color', 'green', 'size', 10);
        end
    end
    figure(3)
    imshow(sceneframe)
    n = n + 1;
end
release(videoPlayer);