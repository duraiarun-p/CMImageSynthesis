clc;clear;close all;
%%

sli=10;
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Perform_I2I_3D_Output/3D_Output_All/run_3/run0/arch/Pred_volumes_alpha.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Perform_I2I_3D_Output/');
% [CT_P,CB]=volthresh(CT,CB);
CT_P(CT_P<0.002)=0;
fig1=figure(1);
% imshow(CT(:,:,sli),[]);
imagesc(CT(:,:,sli));colormap(gray);
saveas(fig1,'3D-CT.png');
fig1=figure(1);
% imshow(CT_P(:,:,sli),[]);
imagesc(CT_P(:,:,sli));colormap(gray);
saveas(fig1,'3D-CT-a.png');
%%
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Perform_I2I_3D_Output/3D_Output_All/run_3/run1/arch/Pred_volumes_beta.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Perform_I2I_3D_Output/');
% [CT,CB]=volthresh(CT,CB);
CT_P(CT_P<0.002)=0;
fig1=figure(1);
% imshow(CT_P(:,:,sli),[]);
imagesc(CT_P(:,:,sli));colormap(gray);
saveas(fig1,'3D-CT-b.png');
%%
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Perform_I2I_3D_Output/3D_Output_All/run_3/run2/arch/Pred_volumes_gamma.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Perform_I2I_3D_Output/');
% [CT,CB]=volthresh(CT,CB);
CT_P(CT_P<0.002)=0;
fig1=figure(1);
% imshow(CT_P(:,:,sli),[]);
imagesc(CT_P(:,:,sli));colormap(gray);
saveas(fig1,'3D-CT-g.png');

%%
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Perform_I2I_3D_Output/3D_Output_All/run_3/run3/arch/Pred_volumes_epsilon.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/3D/Perf_Comp_3D/Perform_I2I_3D_Output/');
% [CT,CB]=volthresh(CT,CB);
CT_P(CT_P<0.002)=0;
fig1=figure(1);
% imshow(CT_P(:,:,sli),[]);
imagesc(CT_P(:,:,sli));colormap(gray);
saveas(fig1,'3D-CT-e.png');

%%
