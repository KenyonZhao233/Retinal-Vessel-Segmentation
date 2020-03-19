function [segImage] = vesselSegPC(inputImage)
%Vessel Extraction of retinal fundus Image using principal curvature
%inputImage : input retinal fundus image
%output - segImage : Segmented binary image 
%Author : Achintha Iroshan ,University of Moratuwa

%Generation of image mask
mask = im2bw(inputImage,20/255);
se = strel('diamond',20);
erodedmask = im2uint8(imerode(mask,se));


%Apply gaussian filter to the image where s=1.45
img3= imgaussfilt(inputImage(:,:,2) ,1.45);

%Finding lamda - principal curvature
lamda2=prinCur(img3);
maxprincv = im2uint8(lamda2/max(lamda2(:)));
maxprincvmsk = maxprincv.*(erodedmask/255);

%Contrast enhancement. 
newprI = adapthisteq(maxprincvmsk,'numTiles',[8 8],'nBins',128);
thresh = isodata(newprI);
vessels = im2bw(newprI,thresh);

%Filtering out small segments
vessels = bwareaopen(vessels, 200);
segImage = vessels;


