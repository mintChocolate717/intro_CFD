
function amplification_factor(fig,fname,tit,Gh,ss)

fs=15;
beta=linspace(0,pi,180);

% loop on ss
for i=1:length(ss)

  [mG,pG,G]=feval(Gh,beta,ss(i)); 
  x=mG.*cos(beta); y=mG.*sin(beta);

  figure(fig(1));
  plot(x,y,'k-');
  h=text(x(end),y(end),sprintf('\\sigma: %3.2f',ss(i)));
  set(h,'FontSize',fs);
  hold on;

  figure(fig(2));
  x=beta; y=mG;
  plot(x,y,'r--');
  h=text(x(end),y(end),sprintf('\\sigma: %3.2f',ss(i)));
  set(h,'FontSize',fs);
  hold on;

  x=pG.*cos(beta); y=pG.*sin(beta);

  figure(fig(3));
  plot(x,y,'k-');
  h=text(x(end),y(end),sprintf('\\sigma: %3.2f',ss(i)));
  set(h,'FontSize',fs);
  hold on;

  figure(fig(4));
  x=beta; y=pG;
  plot(x,y,'k-');
  h=text(x(end),y(end),sprintf('\\sigma: %3.2f',ss(i)));
  set(h,'FontSize',fs);
  hold on;

end

% add unit circle
figure(fig(1));
x=cos(beta); y=sin(beta);
plot(x,y,'k--'); hold off

axis equal; axis([-1.5,1.5,0,1.5]);
xticks([-1.5:0.5:1.5]);
xticklabels({'1.5','1','0.5','0','0.5','1','1.5'});
title(sprintf('%s (relative magnitude error)',tit));
set(gca,'FontSize',fs);

figure(fig(2));
x=beta; y=ones(size(mG));
plot(x,y,'k-'); hold off;
v=axis; axis([0,pi,v(3:4)]);
xticks(pi.*[0:0.25:1]);
xticklabels({'0','\pi/4','\pi/2','3\pi/4','\pi'});
title(sprintf('%s (relative magnitude error)',tit));
xlabel('Phase angle \beta');
set(gca,'FontSize',fs);

% add unit circle
figure(fig(3));
x=cos(beta); y=sin(beta);
plot(x,y,'k--'); hold off

axis equal; axis([-1.5,1.5,0,1.5]);
xticks([-1.5:0.5:1.5]);
xticklabels({'1.5','1','0.5','0','0.5','1','1.5'});
title(sprintf('%s (relative phase error)',tit));
set(gca,'FontSize',fs);

figure(fig(4));
x=beta; y=ones(size(pG));
plot(x,y,'k-'); hold off;
v=axis; axis([0,pi,v(3:4)]);
xticks(pi.*[0:0.25:1]);
xticklabels({'0','\pi/4','\pi/2','3\pi/4','\pi'});
title(sprintf('%s (relative phase error)',tit));
xlabel('Phase angle \beta');
set(gca,'FontSize',fs);
