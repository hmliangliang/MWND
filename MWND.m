function Result = MWND( data1,k )
%���㷨��Ҫִ��������ĵ�xinsuanf
%data: ���������,ÿһ��Ϊһ������   k:����������Ŀ
lambda=5.5;%��ʼ������
tt=2;
data=cell(1,4);%����ÿһ������������
no_dims=round(3/4*size(data1,2));%ȷ����ά��ά��
data{1}= pca(data1,no_dims);%PCA�㷨��ά�㷨��ά�㷨
data{2}=(lle(data1',8*k,no_dims))';%LLE�㷨��ά�㷨
data{3} = Laplacianf(data1, 8*k, no_dims);%Laplacian����ӳ�併ά�㷨
data{4} = nnmf(data1,no_dims);%�Ǹ�����ֽ⽵ά�㷨
w=zeros(4,no_dims);%��ʼ��ÿ������ÿ�����Ե�Ȩֵ,ÿһ�д���һ�����ݼ�,ÿһ�д���һ������
C=cell(1,4);%����ÿһ���������ľ�����
MSE=zeros(1,4);%����ÿһ�ξ�������MSE
for i=1:4
    for j=1:no_dims
        w(i,j)=std(data{i}(:,j));%���㷽��
    end
end
%Ȩֵ��һ��
for i=1:4
    for j=1:no_dims
        w(i,j)=w(i,j)/sum(w(i,:));%����Ȩֵ
    end
end
N=floor(1/2*size(data1,1));%����ȡ������Ŀ
delta=zeros(1,4);%����deltaֵ
for i=1:4
    sample=data{i}(randperm(size(data{i},1),N),:);
    dis=0;
    for j=1:size(sample,1)
        for jj=j+1:size(sample,1)
            dis=dis+norm(sample(j,:)-sample(jj,:),2);
        end
    end
    delta(i)=tt*sqrt(1/(N*(N-1))*dis);%�����i�����ݼ�������뾶
end
p=cell(1,4);%����ÿһ���������������ĵ�
for i=1:4%��ʼ���ĵ����ò���������
   p{i}=Selection(data{i}, k,delta(i),lambda);%ѡ���ʼ���ĵ�
   %p{i}=data{i}(randperm( size(data{i},1),k),:);
end
%��ʼ���ӿռ����
N_MAX=10;%���ĵ�������
beta=2;
e=0.001;%MSE���������
alpha=2;%����ԭ����
for ij=1:4%��ÿһ���������ݽ��е����ؾ���
    mydata=data{ij};
    for j=1:size(mydata,2)
        maxnum=max(mydata(:,j));
        minnum=min(mydata(:,j));
        for i=1:size(mydata,1)
            mydata(i,j)=(mydata(i,j)-minnum)/(maxnum-minnum);
        end
    end
    num=0;%��¼��ǰ�����Ĵ���
    mse=inf;%��¼�������ľ������
    W=1/size(mydata,2)*ones(k,size(mydata,2));
    while mse>e%�㷨��δ����
        num=num+1;%�����Ĵ�����1
        if num>N_MAX%���������ﵽ���
            break;
        end
        %���»��־���U
        U=zeros(size(mydata,1),k);%��ʼ�����־���
        d=zeros(size(mydata,1),k);%���㵱ǰ������������صľ���
        for i=1:size(mydata,1)
            for j=1:k
                d(i,j)=Distance(mydata(i,:),p{ij}(j,:),w(ij,:),W(j,:));
            end
        end
        clear i;
        [~,s]=min(d,[],2);%ȷ����صĹ���,sΪ��������صı��
        for i=1:size(U,1)%���»��־����ֵ
            U(i,s(i,1))=1;%ȷ�������Ĺ���
        end
        %���¾�������ĵ�
        clear i;
        for i=1:k
            s=find(U(:,i)==1);%�ҳ�ÿһ����ص�������
            if isempty(s)==0%����ص������㲻Ϊ��
                ss=mydata(s,:);
                p{ij}(i,:)=mean(mydata(s,:));%��ȡ�����µ����ĵ�
            end
        end
        %����Ȩֵ�ӿռ������ȨֵW
        clear i;
        for i=1:k
            d=zeros(1,size(mydata,2));%���浱ǰ����ڸ��������ϵ�Ȩֵ
            s=find(U(:,i)==1);%�ҳ���ǰ��ص�������
            for j=1:size(mydata,2)
                for r=1:size(s,1)
                  d(j)=d(j)+w(ij,j)*(mydata(s(r,1),j)-p{ij}(i,j))^2;
                end
            end
            %����ָ������
            for j=1:size(mydata,2)
                d(j)=exp(-d(j)/alpha);
            end
            %���㵱ǰ�����ڸ��������ϵ�Ȩֵ,��Ȩֵ��һ��
            for j=1:size(mydata,2)
                W(i,j)=d(j)/sum(d);%�����ӿռ��ϵ�ÿһ��������Ȩֵ
            end
       end
        %����ȫ������Ȩֵw
        d=zeros(1,size(mydata,2));%����������
        clear i;
        clear j;
        for j=1:size(mydata,2)
            for i=1:size(mydata,1)
                cnum=find(U(i,:)==1);%���ҵ�ǰ��صı��
                d(j)=d(j)+W(cnum,j)*(mydata(i,j)-p{ij}(cnum,j))^2;%���㵱ǰ�������ڵ�ǰ����ά���ϵľ���
            end
        end
        clear i;
        clear j;
        for j=1:size(mydata,2)%����ָ������
            d(j)=exp(-d(j)/beta);
        end
        clear j;
        for j=1:size(mydata,2)%�������յ�Ȩֵ,��Ȩֵ�Ĺ�һ��
            w(j)=d(j)/sum(d);
        end
        %�����������MSE
        mse=0;
        clear i;
        clear j;
        for i=1:size(mydata,1)%����������
            s=find(U(i,:)==1);%���㵱ǰ��������ر��
            mse=mse+norm(mydata(i,:)-p{ij}(s,:),2);
        end
        mse=1/size(mydata,1)*sqrt(mse);%���㵱ǰ�������ľ������
    end
    clear i;
    for i=1:k%��ȡÿһ����ص��������
        C{ij}{i}=(find(U(:,i)==1))';
    end
    MSE(ij)=mse;
end
%���²�����Ҫ���ں϶��������
cw=zeros(1,4);%��Ҫ�Ǳ����������Ȩֵ
for i=1:4
    cw(i)=MSE(i)/sum(MSE)+Granularity(C{i});
end
%Ȩֵ���й�һ��
for i=1:4
    cw(i)=exp(-cw(i));
end
dd=sum(cw);
for i=1:4
    cw(i)=cw(i)/dd;
end
M=zeros(size(data1,1),size(data1,1));%��ʼ��Ȩֵ����
for i=1:4
    CR=C{i};%��ȡ����Ľ��
    for x=1:k
        cdata=CR{x};
        if isempty(cdata)==0
            for y=1:size(cdata,2)
                for z=y+1:size(cdata,2)
                    M(cdata(y),cdata(z))=M(cdata(y),cdata(z))+cw(i);
                end
            end
        end
    end
end
for i=1:size(M,1)
    for j=i+1:size(M,2)
        M(j,i)=M(i,j);
    end
end
Result=merged(M,k);%���ò�ξ���˼��,�ϲ����ƵĶ���
end

function C=merged(M,k)
SM=M;%��ֹM��Ԫ���⵽�ƻ�
C=cell(1,size(M,1));%������������ݵ����
for i=1:size(M,1)%��ʼ��C
    C{i}=i;
end
while size(C,2)>k%�ϲ���������Ŀ����k
    [m,n]=MAX_NUM(SM);%�����ֵ��Ԫ�����ڵ��к���
    if m>=n%m��֤Ϊ��С�����
      t=m;
      m=n;
      n=t;
    end
    SM([m,n],:)=[];%ɾ����
    %SM(n-1,:)=[];%ɾ����
    SM(:,[m,n])=[];%ɾ����
    %SM(:,n-1)=[];%ɾ����
    temp=[C{m},C{n}];
    C([m,n])=[];
    %C(n-1)=[];
    C=[C,temp];
    %���һ�к�һ��
    SM=[SM,zeros(size(SM,1),1)];
    SM=[SM;zeros(1,size(SM,2))];
    col=size(SM,2);
    for i=1:size(SM,1)%���¼����е�Ԫ��
        if size(C{i},2)>1%����C{i}��һ�����
            for x=1:size(temp,2)
                for y=1:size(C{i},2)
                    if i==col
                        SM(i,col)=0;
                    else
                       SM(i,col)=SM(i,col)+M(C{i}(y),temp(x));
                    end
                end
            end
            SM(i,col)=1/size(temp,2)*(1/size(C{i},2))*SM(i,col);
            SM(col,i)=SM(i,col);
        else%����C{i}��һ�����ݵ�
            for x=1:size(temp,2)
                if i==col
                    SM(i,col)=0;
                else
                   SM(i,col)=SM(i,col)+M(temp(x),C{i});
                end
            end
            SM(i,col)=1/size(temp,2)*SM(i,col);
            SM(col,i)=SM(i,col);
        end
    end
    clear temp;
end
for i=1:size(C,2)
    C{i}=unique(C{i});
end
end

%Ѱ�Ҿ���M�е����ֵ,������һ��ֵ�ı��
function [m,n]=MAX_NUM(M)
%����:MΪһ����ֵ����
%���: mΪ���ֵԪ�����ڵ��к�  nΪ���ֵԪ�����ڵ��к�
d=max(max(M));%Ѱ��M���������ֵԪ��
for i=1:size(M,1)
    for j=1:size(M,2)
        if M(i,j)==d%Ѱ�����ֵԪ�ص�λ��
            m=i;
            n=j;
        end
    end
end
end

%���������������
function g=Granularity(C)
%�˺�����Ҫ�������������Ϣ����
%data:��Ҫ������  C:����Ľ��,���ݽṹΪcell��
g=0;
s=0;
for i=1:size(C,2)
    s=s+length(C{i});%�������ݵ�����
end
for i=1:size(C,2)
    g=g+1/s^2*(length(C{i})^2/s^2);%�������ݵ�����
end 
end

%����������Ȩ��ŷʽ�����ƽ��
function d=Distance(data1,data2,w,W)
%���� data:��������,ÿһ�д���һ������     W:�����ӿռ��Ȩֵ���� 
%w:����ԭ���ݵ�����Ȩֵ
d=0;
for i=1:size(data1,2)
    d=d+w(i)*W(i)*(data1(i)-data2(i))^2;
end
end 

%ѡ��k����ʼ���ĵ�
function p=Selection(data, k,delta,lambda)%ѡ��k����ʼ���ĵ�
%����   delta:����뾶  lambda:����  k:��������ĵ����Ŀ
%���   pΪk�������
p=[];%����ʼ������Ϊ��
%����ÿ�����������
Ne=zeros(size(data,1),size(data,1));%�����������
for i=1:size(data,1)
    for j=1:size(data,1)
        if norm(data(i,:)-data(j,:),2)<=delta%��ǰ����λ��������
            Ne(i,j)=1;
        end
    end
end
density=zeros(1,size(data,1));%����ÿ������������ܶ�
for i=1:size(data,1)
    for j=1:size(data,1)
        if Ne(i,j)==1%˵������jΪ����i��������
            if all(ismember(find(Ne(j,:)==1),intersect(find(Ne(i,:)==1),find(Ne(j,:)==1))))==1%˵������jλ�ڶ���i���½��Ƽ���
                density(i)=density(i)+lambda;
            elseif isempty(intersect(find(Ne(i,:)==1),find(Ne(j,:)==1)))==0%˵������jλ�ڶ���i���Ͻ��Ƽ���
                density(i)=density(i)+lambda*(length(intersect(find(Ne(i,:)==1),find(Ne(j,:)==1)))/length(find(Ne(j,:)==1)));
            end
        end
    end
end
[~,s]=sort(density,'descend');%�Զ�����ܶȽ��н�������,s������ǽ������е����
for i=1:size(s,1)-1
    if norm(data(i,:)-data(i+1,:),2)>=3*delta%���ڵ�������֮��ľ������Ҫ��
        p=[p;data(i,:)];%�Ѵ˵���뵽ѡ������ĵ���ȥ
    end
end
if size(p,1)<k%��������Ҫ������ݵ�����k��
    p=[p;data(randperm( size(data,1),k-size(p,1)),:)];%ʣ�µ����ݵ����ѡ��
end
end
%������˹������ά�㷨
function mydata=Laplacianf(data, k, no_dims)
%data:��Ҫ��ά������,ÿһ�д���һ������  k:������ڵ���Ŀ  no_dims:��ά�����ݵ�ά��
%mydata:��ά�����������,ÿһ�д���һ������
N=zeros(size(data,1),size(data,1));%�����Ƿ���k����,����k-��������ֵΪ1,����Ϊ0
d=zeros(size(data,1),size(data,1));%������ÿ������֮��ľ���
for i=1:size(data,1)
    for j=i:size(data,1)
        d(i,j)=norm(data(i,:)-data(j,:),2);
        d(j,i)=norm(data(i,:)-data(j,:),2);
    end
end
for i=1:size(data,1)
    [~,s]=sort(d(i,:));%Ѱ��ÿ�������k����
    N(i,s(1:k+1))=1;
    N(s(1:k+1),i)=1;
end
%����Ȩֵ
W=zeros(size(data,1),size(data,1));%Ȩֵ����
for i=1:size(W,1)%����Ȩֵ����
    for j=i+1:size(W,2)
        if N(i,j)>0
           W(i,j)=exp(-d(i,j)/5);
           W(j,i)=exp(-d(i,j)/5);
        end
    end
end
for i=1:size(W,1)%����Ȩֵ����
    for j=i+1:size(W,2)
        W(i,j)=W(i,j)/sum(W(i,:));
        W(j,i)=W(i,j);
    end
end
D=zeros(size(data,1),size(data,1));%ͼ�ĶȾ���
for i=1:size(data,1)%��ȡ�Ⱦ���D�ĶԽ�Ԫ��ֵ
    D(i,i)=sum(W(i,:));
end
L=D-W;%����ͼ��������˹����
[V,d]= eigs(L, no_dims+1,'SM');%����������˹������������ֵD,VΪ��Ӧ����ֵ����������
[~,ins]=sort(diag(d));
ins=ins';
mydata=V(:,ins(2:no_dims+1));%��ȡ���յķֽ���,��ȥ����ֵ��С����Ӧ����������
end

%LLE�㷨��ά
function [Y] = lle(X,K,d)
% LLE ALGORITHM (using K nearest neighbors)
% [Y] = lle(X,K,dmax)
% X    ��data as D x N matrix (D = dimensionality, N = #points)
% K    ��number of neighbors
% dmax ��max embedding dimensionality
% Y    ��embedding as dmax x N matrix
[D,N] = size(X);
fprintf(1,'LLE running on %d points in %d dimensions\n',N,D);
% Step1: compute pairwise distances & find neighbour
fprintf(1,'-->Finding %d nearest neighbours.\n',K);
X2 = sum(X.^2,1);
distance = repmat(X2,N,1)+repmat(X2',1,N)-2*X'*X;
[sorted,index] = sort(distance);
neighborhood = index(2:(1+K),:);
% Step2: solve for recinstruction weights
fprintf(1,'-->Solving for reconstruction weights.\n');
if(K>D)
  fprintf(1,'   [note: K>D; regularization will be used]\n');
  tol=1e-3; % regularlizer in case constrained fits are ill conditioned
else
  tol=0;
end
W = zeros(K,N);
for ii=1:N
   z = X(:,neighborhood(:,ii))-repmat(X(:,ii),1,K); % shift ith pt to origin
   C = z'*z;                                        % local covariance
   C = C + eye(K,K)*tol*trace(C);                   % regularlization (K>D)
   W(:,ii) = C\ones(K,1);                           % solve Cw=1
   W(:,ii) = W(:,ii)/sum(W(:,ii));                  % enforce sum(w)=1
end
% Step 3: compute embedding from eigenvects of cost matrix M=(I-W)'(I-W)
fprintf(1,'-->Computing embedding.\n');
% M=eye(N,N); % use a sparse matrix with storage for 4KN nonzero elements
M = sparse(1:N,1:N,ones(1,N),N,N,4*K*N);
for ii=1:N
   w = W(:,ii);
   jj = neighborhood(:,ii);
   M(ii,jj) = M(ii,jj) - w';
   M(jj,ii) = M(jj,ii) - w;
   M(jj,jj) = M(jj,jj) + w*w';
end
% calculation of embedding
options.disp = 0;
options.isreal = 1;
options.issym = 1;
[Y,eigenvals] = eigs(M,d+1,0,options);
Y = Y(:,2:d+1)'*sqrt(N); % bottom evect is [1,1,1,1...] with eval 0
end