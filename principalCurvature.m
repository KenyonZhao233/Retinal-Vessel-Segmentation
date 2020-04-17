function [segImage] = principalCurvature(inputImage,plot)
%Vessel Extraction of retinal fundus Image using principal curvature
%inputImage : input retinal fundus image
%output - segImage : Segmented binary image 
%Reference : Achintha Iroshan ,University of Moratuwa

%Generation of image mask
mask = im2bw(inputImage,40/255);
se = strel('diamond',20);               %str. element of dimond type of size 20 (errosion)
erodedmask = im2uint8(imerode(mask,se));    %erroded img
% figure;
% imshow(erodedmask);
% title("Generation of image mask");

%Apply gaussian filter to the image where s=1.45
img3= imgaussfilt(inputImage(:,:,2) ,1.45);     %sigma st. deviation (s)
% figure;
% imshow(img3);
% title("After Gaussian Filter");

%Finding lamda - principal curvature
lamda2=prinCur(img3);
maxprincv = im2uint8(lamda2/max(lamda2(:)));
maxprincvmsk = maxprincv.*(erodedmask/255);
% figure;
% imshow(maxprincvmsk);
% title("Finding lamda - principal curvature");

%Contrast enhancement. 
newprI = adapthisteq(maxprincvmsk,'numTiles',[8 8],'nBins',128);
thresh = isodata(newprI);
vessels = im2bw(newprI,thresh);
% figure;
% imshow(vessels);
% title("Contrast enhancement.");

%Filtering out small segments
vessels = bwareaopen(vessels, 200);
segImage = vessels;
% figure;
% imshow(segImage);
% title("Filtering out small segments");
if plot == true
        figure();        
        subplot(2,3,2);imshow(inputImage(:,:,2),[]);title('绿色通道');
        subplot(2,3,1);imshow(erodedmask,[]);title('掩模');
        subplot(2,3,3);imshow(img3,[]);title('高斯滤波后');
        subplot(2,3,4);imshow(maxprincvmsk,[]);title('主曲率');
        subplot(2,3,5);imshow(vessels,[]);title('对比度增强');
        subplot(2,3,6);imshow(vessels,[]);title('滤波细血管');
end
end

