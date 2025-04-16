close all
clear all
clc

%- fig=[1,2,3,4];
%- fname='fou';
%- Gh=@fou;
%- ss=[0.25,0.5,0.75,1.0];
%- title='First-order Upwind';
%- amplification_factor(fig,fname,title,Gh,ss);

%- fig=[1,2,3,4];
%- fname='implicit';
%- Gh=@implicit;
%- ss=[0.25,0.5,0.75,1.0,1.5];
%- title='Implicit Centered';
%- amplification_factor(fig,fname,title,Gh,ss);

fig=[1,2,3,4];
fname='lax-friedrichs';
Gh=@laxfriedrichs;
ss=[0.25,0.5,0.75,1.0];
title='Lax-Friedrichs';
amplification_factor(fig,fname,title,Gh,ss);

%- fig=[1,2,3,4];
%- fname='beam-warming';
%- Gh=@beamwarming;
%- ss=[0.25,0.5,0.75,1.0,2.0,2.05];
%- title='Beam-Warming';
%- amplification_factor(fig,fname,title,Gh,ss);

%- fig=[1,2,3,4];
%- fname='lax-wendroff';
%- Gh=@laxwendroff;
%- ss=[0.25,0.5,0.75,1.0];
%- title='Lax-Wendroff';
%- amplification_factor(fig,fname,title,Gh,ss);







