clc;clear;close all;
%%

sli=32;
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/Perform_I2I_2D_Output/run/run0/arch/Pred_volumes_alpha.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/');
% [CT_P,CB]=volthresh(CT,CB);
CB_P(CB_P<0.002)=0;
fig1=figure(1);
% imshow(CT(:,:,sli),[]);
imagesc(CB(:,:,sli));colormap(gray);
saveas(fig1,'2D-CB.png');
fig1=figure(1);
% imshow(CB_P(:,:,sli),[]);
imagesc(CB_P(:,:,sli));colormap(gray);
saveas(fig1,'2D-CB-a.png');
%%
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/Perform_I2I_2D_Output/run/run1/arch/Pred_volumes_beta.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/');
% [CB,CB]=volthresh(CB,CB);
CB_P(CB_P<0.002)=0;
fig1=figure(1);
% imshow(CB_P(:,:,sli),[]);
imagesc(CB_P(:,:,sli));colormap(gray);
saveas(fig1,'2D-CB-b.png');
%%
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/Perform_I2I_2D_Output/run/run2/arch/Pred_volumes_gamma.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/');
% [CB,CB]=volthresh(CB,CB);
CB_P(CB_P<0.002)=0;
fig1=figure(1);
% imshow(CB_P(:,:,sli),[]);
imagesc(CB_P(:,:,sli));colormap(gray);
saveas(fig1,'2D-CB-g.png');
%%
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/Perform_I2I_2D_Output/run/run3/arch/Pred_volumes_delta.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/');
% [CB,CB]=volthresh(CB,CB);
CB_P(CB_P<0.002)=0;
fig1=figure(1);
% imshow(CB_P(:,:,sli),[]);
imagesc(CB_P(:,:,sli));colormap(gray);
saveas(fig1,'2D-CB-d.png');

%%
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/Perform_I2I_2D_Output/run/run4/arch/Pred_volumes_epsilon.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/');
% [CB,CB]=volthresh(CB,CB);
CB_P(CB_P<0.002)=0;
fig1=figure(1);
% imshow(CB_P(:,:,sli),[]);
imagesc(CB_P(:,:,sli));colormap(gray);
saveas(fig1,'2D-CB-e.png');

%%
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/Perform_I2I_2D_Output/run/run5/arch/Pred_volumes_zeta.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/');
% [CB,CB]=volthresh(CB,CB);
CB_P(CB_P<0.002)=0;
fig1=figure(1);
% imshow(CB_P(:,:,sli),[]);
imagesc(CB_P(:,:,sli));colormap(gray);
saveas(fig1,'2D-CB-z.png');
%%
load('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/Perform_I2I_2D_Output/run/run6/arch/Pred_volumes_eta.mat');
cd('/home/arun/Documents/PyWSPrecision/CMImageSynthesis/2D/Perf_Comp_2D/');
% [CB,CB]=volthresh(CB,CB);
CB_P(CB_P<0.002)=0;
fig1=figure(1);
% imshow(CB_P(:,:,sli),[]);
imagesc(CB_P(:,:,sli));colormap(gray);
saveas(fig1,'2D-CB-et.png');
%%
function [CT]=volthresh(CT)
CT(CT<0.002)=0;
% CB(CB<0.002)=0;
end
