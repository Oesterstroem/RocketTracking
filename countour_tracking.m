%% Tracking using centroids and edge
clear
close all

% Output video player
videoOut = vision.VideoPlayer('Name', 'Rocket Tracking', ...
    'Position', [640 480 640 480]);

% Read Video and make reference image
videoReader = vision.VideoFileReader('am_rocket.MOV');
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
% Remove Noise and find edge
canny_img = edge(img_gray,'canny',0.3);
% Dilate image
se = strel('line',11,90);
dilate_img =  imdilate(canny_img,se);
img_gray  = dilate_img;
Ilabel = logical(canny_img);
stat = regionprops(Ilabel,'centroid');
% Display grayscale image
figure,
imshow(img_gray);
title('Centroid of Rocket Image');
hold on;

plot(stat(1).Centroid(1),stat(1).Centroid(2),'ro')
% Determine frame size
frame = step(videoReader);
frameSize = size(frame);
videoPlayer = vision.VideoPlayer('Position', [100 100 [frameSize(2), frameSize(1)]+30]);
% Init bin image
bin_img = 0;
% Initial display values
position = [0 0];
box_color = {'green'};
text_str = sprintf('Position: %0.2f %0.2f ', stat(1).Centroid(1),stat(1).Centroid(2));
% Loop through image
for n=1:800
    tic
    origFrame = step(videoReader);
    % Grayscale image
    grayframe = rgb2gray(origFrame);
    % Convert to Binary
    bin_img = im2bw(grayframe,0.4);
    % Dilate binary image
    se = strel('octagon',6);
    % Dilate binary image
    dil_frame =  imdilate(~bin_img, se);
    Ilabel = logical(dil_frame);
    % Determine centroids
    stat = regionprops(Ilabel,'centroid');
    % If fail, use edge detection on grayscale
    try
        xyPoints = round([stat(1).Centroid(1) stat(1).Centroid(2) 2]);
    catch
        % Edge detection
        edgeframe = edge(grayframe,'Canny',0.3);
        [x,y] = find(edgeframe,1);
%         row = floor(tmp(1)/frameSize(1));
%         col = mod(tmp(1),frameSize(2));
        
        xyPoints = [x y 2];
        figure(14);
        imshow(edgeframe)
    end
    text_str = sprintf('Position: %0.2f %0.2f ', xyPoints(1),xyPoints(2));
    origFrame = insertShape(origFrame, 'Circle', xyPoints, 'LineWidth', 2, 'Color', 'red');
    origFrame = insertText(origFrame,position,text_str,...
        'FontSize',18,'BoxColor',box_color,'BoxOpacity',0.4,'TextColor','green');
    figure(15);
    imshow(origFrame)
    toc
end
release(videoPlayer);

