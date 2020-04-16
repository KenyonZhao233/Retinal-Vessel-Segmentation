addpath('others');

im = imread('DIP-Project2-STARE//stare-images//im0001.ppm');

figure();
subplot(1,5,1);imshow(im);t2=clock;title('original');
subplot(1,5,2);imshow(matchedFilter(im));t1=t2;t2=clock;title(['matchedFilter:',num2str(etime(t2,t1)),'s']);
subplot(1,5,3);imshow(gaussDerivativeFilter(im));t1=t2;t2=clock;title(['gaussDerivativeFilter:',num2str(etime(t2,t1)),'s']);
subplot(1,5,4);imshow(laplacianPyramids(im));t1=t2;t2=clock;title(['laplacianPyramids:',num2str(etime(t2,t1)),'s']);
subplot(1,5,5);imshow(principalCurvature(im));t1=t2;t2=clock;title(['principalCurvature:',num2str(etime(t2,t1)),'s']);

