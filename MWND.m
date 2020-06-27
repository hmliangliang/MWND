function Result = MWND( data1,k )
%此算法主要执行所提出的的xinsuanf
%data: 输入的数据,每一行为一个样本   k:聚类的类簇数目
lambda=5.5;%初始化参数
tt=2;
data=cell(1,4);%保存每一个特征的数据
no_dims=round(3/4*size(data1,2));%确定降维的维数
data{1}= pca(data1,no_dims);%PCA算法降维算法降维算法
data{2}=(lle(data1',8*k,no_dims))';%LLE算法降维算法
data{3} = Laplacianf(data1, 8*k, no_dims);%Laplacian特征映射降维算法
data{4} = nnmf(data1,no_dims);%非负矩阵分解降维算法
w=zeros(4,no_dims);%初始化每个特征每个属性的权值,每一行代表一个数据集,每一列代表一个特征
C=cell(1,4);%保存每一个特征集的聚类结果
MSE=zeros(1,4);%保存每一次聚类结果的MSE
for i=1:4
    for j=1:no_dims
        w(i,j)=std(data{i}(:,j));%计算方差
    end
end
%权值归一化
for i=1:4
    for j=1:no_dims
        w(i,j)=w(i,j)/sum(w(i,:));%计算权值
    end
end
N=floor(1/2*size(data1,1));%计算取样额数目
delta=zeros(1,4);%计算delta值
for i=1:4
    sample=data{i}(randperm(size(data{i},1),N),:);
    dis=0;
    for j=1:size(sample,1)
        for jj=j+1:size(sample,1)
            dis=dis+norm(sample(j,:)-sample(jj,:),2);
        end
    end
    delta(i)=tt*sqrt(1/(N*(N-1))*dis);%计算第i个数据集的邻域半径
end
p=cell(1,4);%保存每一个特征的类内中心点
for i=1:4%初始中心点设置部分在这里
   p{i}=Selection(data{i}, k,delta(i),lambda);%选择初始中心点
   %p{i}=data{i}(randperm( size(data{i},1),k),:);
end
%开始软子空间聚类
N_MAX=10;%最大的迭代次数
beta=2;
e=0.001;%MSE的误差上限
alpha=2;%参数原文中
for ij=1:4%对每一个特征数据进行单独地聚类
    mydata=data{ij};
    for j=1:size(mydata,2)
        maxnum=max(mydata(:,j));
        minnum=min(mydata(:,j));
        for i=1:size(mydata,1)
            mydata(i,j)=(mydata(i,j)-minnum)/(maxnum-minnum);
        end
    end
    num=0;%记录当前迭代的次数
    mse=inf;%记录聚类结果的均方误差
    W=1/size(mydata,2)*ones(k,size(mydata,2));
    while mse>e%算法还未收敛
        num=num+1;%迭代的次数加1
        if num>N_MAX%迭代次数达到最大
            break;
        end
        %更新划分矩阵U
        U=zeros(size(mydata,1),k);%初始化划分矩阵
        d=zeros(size(mydata,1),k);%计算当前样本到各个类簇的距离
        for i=1:size(mydata,1)
            for j=1:k
                d(i,j)=Distance(mydata(i,:),p{ij}(j,:),w(ij,:),W(j,:));
            end
        end
        clear i;
        [~,s]=min(d,[],2);%确定类簇的归属,s为归属的类簇的标号
        for i=1:size(U,1)%更新划分矩阵的值
            U(i,s(i,1))=1;%确定样本的归属
        end
        %更新聚类的中心点
        clear i;
        for i=1:k
            s=find(U(:,i)==1);%找出每一个类簇的样本点
            if isempty(s)==0%该类簇的样本点不为空
                ss=mydata(s,:);
                p{ij}(i,:)=mean(mydata(s,:));%获取聚类新的中心点
            end
        end
        %更新权值子空间的特征权值W
        clear i;
        for i=1:k
            d=zeros(1,size(mydata,2));%保存当前类簇在各个特征上的权值
            s=find(U(:,i)==1);%找出当前类簇的样本点
            for j=1:size(mydata,2)
                for r=1:size(s,1)
                  d(j)=d(j)+w(ij,j)*(mydata(s(r,1),j)-p{ij}(i,j))^2;
                end
            end
            %计算指数距离
            for j=1:size(mydata,2)
                d(j)=exp(-d(j)/alpha);
            end
            %计算当前样本在各个特征上的权值,即权值归一化
            for j=1:size(mydata,2)
                W(i,j)=d(j)/sum(d);%更新子空间上的每一个特征的权值
            end
       end
        %更新全局特征权值w
        d=zeros(1,size(mydata,2));%保存距离矩阵
        clear i;
        clear j;
        for j=1:size(mydata,2)
            for i=1:size(mydata,1)
                cnum=find(U(i,:)==1);%查找当前类簇的标号
                d(j)=d(j)+W(cnum,j)*(mydata(i,j)-p{ij}(cnum,j))^2;%计算当前样本点在当前特征维度上的距离
            end
        end
        clear i;
        clear j;
        for j=1:size(mydata,2)%计算指数距离
            d(j)=exp(-d(j)/beta);
        end
        clear j;
        for j=1:size(mydata,2)%计算最终的权值,即权值的归一化
            w(j)=d(j)/sum(d);
        end
        %计算聚类结果的MSE
        mse=0;
        clear i;
        clear j;
        for i=1:size(mydata,1)%计算均方误差
            s=find(U(i,:)==1);%计算当前样本的类簇标号
            mse=mse+norm(mydata(i,:)-p{ij}(s,:),2);
        end
        mse=1/size(mydata,1)*sqrt(mse);%计算当前聚类结果的均方误差
    end
    clear i;
    for i=1:k%获取每一个类簇的样本序号
        C{ij}{i}=(find(U(:,i)==1))';
    end
    MSE(ij)=mse;
end
%以下部分主要是融合多个聚类结果
cw=zeros(1,4);%主要是保存聚类结果的权值
for i=1:4
    cw(i)=MSE(i)/sum(MSE)+Granularity(C{i});
end
%权值进行归一化
for i=1:4
    cw(i)=exp(-cw(i));
end
dd=sum(cw);
for i=1:4
    cw(i)=cw(i)/dd;
end
M=zeros(size(data1,1),size(data1,1));%初始化权值矩阵
for i=1:4
    CR=C{i};%读取聚类的结果
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
Result=merged(M,k);%采用层次聚类思想,合并相似的对象
end

function C=merged(M,k)
SM=M;%防止M中元素遭到破坏
C=cell(1,size(M,1));%保存聚类结果数据的序号
for i=1:size(M,1)%初始化C
    C{i}=i;
end
while size(C,2)>k%合并后的类簇数目大于k
    [m,n]=MAX_NUM(SM);%找最大值的元素所在的行和列
    if m>=n%m保证为较小的序号
      t=m;
      m=n;
      n=t;
    end
    SM([m,n],:)=[];%删除行
    %SM(n-1,:)=[];%删除行
    SM(:,[m,n])=[];%删除列
    %SM(:,n-1)=[];%删除列
    temp=[C{m},C{n}];
    C([m,n])=[];
    %C(n-1)=[];
    C=[C,temp];
    %添加一列和一行
    SM=[SM,zeros(size(SM,1),1)];
    SM=[SM;zeros(1,size(SM,2))];
    col=size(SM,2);
    for i=1:size(SM,1)%更新加入列的元素
        if size(C{i},2)>1%表明C{i}是一个类簇
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
        else%表明C{i}是一个数据点
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

%寻找矩阵M中的最大值,并返回一个值的标号
function [m,n]=MAX_NUM(M)
%输入:M为一个数值矩阵
%输出: m为最大值元素所在的行号  n为最大值元素所在的列号
d=max(max(M));%寻找M矩阵中最大值元素
for i=1:size(M,1)
    for j=1:size(M,2)
        if M(i,j)==d%寻找最大值元素的位置
            m=i;
            n=j;
        end
    end
end
end

%计算聚类结果的粒度
function g=Granularity(C)
%此函数主要计算聚类结果的信息粒度
%data:主要是数据  C:聚类的结果,数据结构为cell型
g=0;
s=0;
for i=1:size(C,2)
    s=s+length(C{i});%计算数据的总数
end
for i=1:size(C,2)
    g=g+1/s^2*(length(C{i})^2/s^2);%计算数据的总数
end 
end

%计算特征加权的欧式距离的平方
function d=Distance(data1,data2,w,W)
%输入 data:数据样本,每一行代表一个样本     W:代表子空间的权值矩阵 
%w:代表原数据的样本权值
d=0;
for i=1:size(data1,2)
    d=d+w(i)*W(i)*(data1(i)-data2(i))^2;
end
end 

%选择k个初始中心点
function p=Selection(data, k,delta,lambda)%选择k个初始中心点
%输入   delta:邻域半径  lambda:参数  k:聚类的中心点的数目
%输出   p为k个对象点
p=[];%将初始矩阵置为空
%计算每个对象的邻域
Ne=zeros(size(data,1),size(data,1));%保存邻域矩阵
for i=1:size(data,1)
    for j=1:size(data,1)
        if norm(data(i,:)-data(j,:),2)<=delta%当前对象位于邻域内
            Ne(i,j)=1;
        end
    end
end
density=zeros(1,size(data,1));%保存每个对象的邻域密度
for i=1:size(data,1)
    for j=1:size(data,1)
        if Ne(i,j)==1%说明对象j为对象i的邻域内
            if all(ismember(find(Ne(j,:)==1),intersect(find(Ne(i,:)==1),find(Ne(j,:)==1))))==1%说明对象j位于对象i的下近似集内
                density(i)=density(i)+lambda;
            elseif isempty(intersect(find(Ne(i,:)==1),find(Ne(j,:)==1)))==0%说明对象j位于对象i的上近似集内
                density(i)=density(i)+lambda*(length(intersect(find(Ne(i,:)==1),find(Ne(j,:)==1)))/length(find(Ne(j,:)==1)));
            end
        end
    end
end
[~,s]=sort(density,'descend');%对对象的密度进行降序排列,s保存的是将序排列的序号
for i=1:size(s,1)-1
    if norm(data(i,:)-data(i+1,:),2)>=3*delta%相邻的两个点之间的距离符合要求
        p=[p;data(i,:)];%把此点加入到选择的中心点中去
    end
end
if size(p,1)<k%表明符合要求的数据点少于k个
    p=[p;data(randperm( size(data,1),k-size(p,1)),:)];%剩下的数据点随机选择
end
end
%拉普拉斯特征降维算法
function mydata=Laplacianf(data, k, no_dims)
%data:需要降维的数据,每一行代表一个对象  k:代表近邻的数目  no_dims:降维后数据的维数
%mydata:降维后输出的数据,每一行代表一个样本
N=zeros(size(data,1),size(data,1));%保存是否是k近邻,若是k-近邻则数值为1,否则为0
d=zeros(size(data,1),size(data,1));%保存与每个对象之间的距离
for i=1:size(data,1)
    for j=i:size(data,1)
        d(i,j)=norm(data(i,:)-data(j,:),2);
        d(j,i)=norm(data(i,:)-data(j,:),2);
    end
end
for i=1:size(data,1)
    [~,s]=sort(d(i,:));%寻找每个对象的k近邻
    N(i,s(1:k+1))=1;
    N(s(1:k+1),i)=1;
end
%计算权值
W=zeros(size(data,1),size(data,1));%权值矩阵
for i=1:size(W,1)%计算权值矩阵
    for j=i+1:size(W,2)
        if N(i,j)>0
           W(i,j)=exp(-d(i,j)/5);
           W(j,i)=exp(-d(i,j)/5);
        end
    end
end
for i=1:size(W,1)%计算权值矩阵
    for j=i+1:size(W,2)
        W(i,j)=W(i,j)/sum(W(i,:));
        W(j,i)=W(i,j);
    end
end
D=zeros(size(data,1),size(data,1));%图的度矩阵
for i=1:size(data,1)%获取度矩阵D的对角元素值
    D(i,i)=sum(W(i,:));
end
L=D-W;%计算图的拉普拉斯矩阵
[V,d]= eigs(L, no_dims+1,'SM');%计算拉普拉斯矩阵所有特征值D,V为对应特征值的特征向量
[~,ins]=sort(diag(d));
ins=ins';
mydata=V(:,ins(2:no_dims+1));%获取最终的分解结果,除去特征值最小所对应的特征向量
end

%LLE算法降维
function [Y] = lle(X,K,d)
% LLE ALGORITHM (using K nearest neighbors)
% [Y] = lle(X,K,dmax)
% X    ：data as D x N matrix (D = dimensionality, N = #points)
% K    ：number of neighbors
% dmax ：max embedding dimensionality
% Y    ：embedding as dmax x N matrix
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