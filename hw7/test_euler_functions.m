clear all
close all
clc

g = 1.4;

load U.mat;

for i=1:size(U,1)
  %W(i,:)=U2W(U(i,:),g);
  %lambda(i,:)=eigsEuler(U(i,:),g);
  %F(i,:)=fluxEuler(U(i,:),g);
end

fprintf('\n\n--- W ----');
for i=1:size(U,1)
  fprintf('\n %12.3e %12.3e %12.3e',W(i,1:3));
end
fprintf('\n');

fprintf('\n\n--- lambda ----');
for i=1:size(U,1)
  fprintf('\n %12.3e %12.3e %12.3e',lambda(i,1:3));
end
fprintf('\n');

fprintf('\n\n--- F ----');
for i=1:size(U,1)
  fprintf('\n %12.3e %12.3e %12.3e',F(i,1:3));
end
fprintf('\n');



