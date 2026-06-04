clc;clear;close all;
%%

sli=32;
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/Perform_I2I_2D_Output/run/run0/arch/Pred_volumes_alpha.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/');
% [CT_P,CB]=volthresh(CT,CB);
CT_P(CT_P<0.002)=0;
fig1=figure(1);
% imshow(CT(:,:,sli),[]);
imagesc(CT(:,:,sli));colormap(gray);
saveas(fig1,'2D-CT.png');
fig1=figure(1);
% imshow(CT_P(:,:,sli),[]);
imagesc(CT_P(:,:,sli));colormap(gray);
saveas(fig1,'2D-CT-a.png');
%%
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/Perform_I2I_2D_Output/run/run1/arch/Pred_volumes_beta.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/');
% [CT,CB]=volthresh(CT,CB);
CT_P(CT_P<0.002)=0;
fig1=figure(1);
% imshow(CT_P(:,:,sli),[]);
imagesc(CT_P(:,:,sli));colormap(gray);
saveas(fig1,'2D-CT-b.png');
%%
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/Perform_I2I_2D_Output/run/run2/arch/Pred_volumes_gamma.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/');
% [CT,CB]=volthresh(CT,CB);
CT_P(CT_P<0.002)=0;
fig1=figure(1);
% imshow(CT_P(:,:,sli),[]);
imagesc(CT_P(:,:,sli));colormap(gray);
saveas(fig1,'2D-CT-g.png');
%%
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/Perform_I2I_2D_Output/run/run3/arch/Pred_volumes_delta.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/');
% [CT,CB]=volthresh(CT,CB);
CT_P(CT_P<0.002)=0;
fig1=figure(1);
% imshow(CT_P(:,:,sli),[]);
imagesc(CT_P(:,:,sli));colormap(gray);
saveas(fig1,'2D-CT-d.png');

%%
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/Perform_I2I_2D_Output/run/run4/arch/Pred_volumes_epsilon.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/');
% [CT,CB]=volthresh(CT,CB);
CT_P(CT_P<0.002)=0;
fig1=figure(1);
% imshow(CT_P(:,:,sli),[]);
imagesc(CT_P(:,:,sli));colormap(gray);
saveas(fig1,'2D-CT-e.png');

%%
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/Perform_I2I_2D_Output/run/run5/arch/Pred_volumes_zeta.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/');
% [CT,CB]=volthresh(CT,CB);
CT_P(CT_P<0.002)=0;
fig1=figure(1);
% imshow(CT_P(:,:,sli),[]);
imagesc(CT_P(:,:,sli));colormap(gray);
saveas(fig1,'2D-CT-z.png');
%%
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/Perform_I2I_2D_Output/run/run6/arch/Pred_volumes_eta.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/');
% [CT,CB]=volthresh(CT,CB);
CT_P(CT_P<0.002)=0;
fig1=figure(1);
% imshow(CT_P(:,:,sli),[]);
imagesc(CT_P(:,:,sli));colormap(gray);
saveas(fig1,'2D-CT-et.png');
%%
function [CT]=volthresh(CT)
CT(CT<0.002)=0;
% CB(CB<0.002)=0;
end
