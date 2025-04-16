function [mG,pG,G]=laxfriedrichs(x,s)
%
% lax-friedrichs 
%

G=cos(x)-i*s*sin(x);
Ge=exp(-i*s*x);
mG=abs(G);
pG=angle(G)./angle(Ge);
