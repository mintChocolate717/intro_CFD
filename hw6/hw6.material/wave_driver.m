clear all
close all
clc

c=1;        % advective speed
L=4*pi;     % computational domain [0,L]
T=2*2*pi;   % end time
M=0;        % intermediate solutions

fexact='exact.dat';

sigma=0.25; % Courant number
n=30;       % number of interior points

%method='forward-upwind';
%method='implicit-central';
%method='beam-warming';
method='lax-wendroff';

% initial conditions
u0 = @(x) sin(x);  % anonymous function

% solve
out=wave_solve(c,L,n,sigma,T,M,u0,method);

% plot
xx=linspace(0,L,1000);
for i=1:size(out.U,2)
  exact(:,i)=u0(xx-out.TT(i))';
  plot(out.x,out.U(:,i),'ko-',...
       xx,u0(xx-out.TT(i)),'r-');
  axis([0,L,-1.1,1.1]);
  xlabel('x');
  ylabel('u(x) and numerical solution');
  title(sprintf('Time is %f',out.TT(i)));
  pause
end

% dump
fout=sprintf('%s_n%g_sigma%f.dat',method,n,sigma);
dlmwrite(fout,[out.x',out.U],'delimiter',' ','precision','%e');
dlmwrite(fexact,[xx',exact],'delimiter',' ','precision','%e');

