function [modes its]=ceemdan(x,Nstd,NR,MaxIter)
x=x(:)';
desvio_x=std(x);
x=x/desvio_x;

modes=zeros(size(x));
temp=zeros(size(x));
aux=zeros(size(x));
acum=zeros(size(x));
iter=zeros(NR,round(log2(length(x))+5));

for i=1:NR
    white_noise{i}=randn(size(x));%creates the noise realizations
end;

for i=1:NR
    modes_white_noise{i}=emd(white_noise{i});%calculates the modes of white gaussian noise
end;

for i=1:NR %calculates the first mode
    temp=x+Nstd*white_noise{i};
    [temp, o, it]=emd(temp,'MAXMODES',1,'MAXITERATIONS',MaxIter);
    temp=temp(1,:);
    aux=aux+temp/NR;
    iter(i,1)=it;
end;

modes=aux; %saves the first mode
k=1;
aux=zeros(size(x));
acum=sum(modes,1);

while  nnz(diff(sign(diff(x-acum))))>2 %calculates the rest of the modes
    for i=1:NR
        tamanio=size(modes_white_noise{i});
        if tamanio(1)>=k+1
            noise=modes_white_noise{i}(k,:);
            noise=noise/std(noise);
            noise=Nstd*noise;
            try
                [temp, o, it]=emd(x-acum+std(x-acum)*noise,'MAXMODES',1,'MAXITERATIONS',MaxIter);
                temp=temp(1,:);
            catch
                it=0;
                temp=x-acum;
            end;
        else
            [temp, o, it]=emd(x-acum,'MAXMODES',1,'MAXITERATIONS',MaxIter);
            temp=temp(1,:);
        end;
        aux=aux+temp/NR;
    iter(i,k+1)=it;    
    end;
    modes=[modes;aux];
    aux=zeros(size(x));
    acum=zeros(size(x));
    acum=sum(modes,1);
    k=k+1;
end;
modes=[modes;(x-acum)];
[a b]=size(modes);
iter=iter(:,1:a);
modes=modes*desvio_x;
its=iter;


   