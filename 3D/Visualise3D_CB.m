clc;clear;close all;
%%

sli=32;
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Perform_I2I_3D_Output/3D_Output_All/run_3/run0/arch/Pred_volumes_alpha.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Perform_I2I_3D_Output/');
% [CT_P,CB]=volthresh(CT,CB);
CB_P(CB_P<0.002)=0;
fig1=figure(1);
% imshow(CB(:,:,sli),[]);
imagesc(CB(:,:,sli));colormap(gray);
saveas(fig1,'3D-CB.png');
fig1=figure(1);
% imshow(CB_P(:,:,sli),[]);
imagesc(CB_P(:,:,sli));colormap(gray);
saveas(fig1,'3D-CB-a.png');
%%
sli=32;
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Perform_I2I_3D_Output/3D_Output_All/run_3/run1/arch/Pred_volumes_beta.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Perform_I2I_3D_Output/');
% [CB,CB]=volthresh(CB,CB);
CB_P(CB_P<0.002)=0;
fig1=figure(1);
% imshow(CB_P(:,:,sli),[]);
imagesc(CB_P(:,:,sli));colormap(gray);
saveas(fig1,'3D-CB-b.png');
%%
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Perform_I2I_3D_Output/3D_Output_All/run_3/run2/arch/Pred_volumes_gamma.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Perform_I2I_3D_Output/');
% [CB,CB]=volthresh(CB,CB);
CB_P(CB_P<0.002)=0;
fig1=figure(1);
% imshow(CB_P(:,:,sli),[]);
imagesc(CB_P(:,:,sli));colormap(gray);
saveas(fig1,'3D-CB-g.png');

%%
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Perform_I2I_3D_Output/3D_Output_All/run_3/run3/arch/Pred_volumes_epsilon.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Perform_I2I_3D_Output/');
% [CB,CB]=volthresh(CB,CB);
CB_P(CB_P<0.002)=0;
fig1=figure(1);
% imshow(CB_P(:,:,sli),[]);
imagesc(CB_P(:,:,sli));colormap(gray);
saveas(fig1,'3D-CB-e.png');

%%
