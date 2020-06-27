tic;
data=waveform;%获取数据集  注：vehicle与magic要先打乱顺序
data =data(randperm(size(data,1)),:);
k=3;%设置聚类类簇的数目
col=size(data,2);%获取数据的列数
mydata=data(:,1:col-1);%获取数据部分
Label=data(:,col);%获取数据的类标签
data=data(:,1:col-1);%获取数据部分
data=zscore(data);%标准化消除量纲的影响
N_MAX=7;%重复执行算法的次数
RR=[];%保存各次运行的R值
FFM=[];%保存各次运行的FM值
PP=[];%保存各次运行的P值
MMSE=[];%保存各次运行的NMI值
NNMI=[];
KK=[];
SSS=[];
qresult=[];
for i=1:N_MAX
%      result=SCIFWSA(data,k);%执行SCIFWSA算法
%      result=FSSCND(data,k);%执行FSSCND算法
%      result=DIFSC( data,k );%执行DIFSC算法
%      result= LRGR(data,k);%执行LRGR算法
    %result=MWND( data,k );%执行MWND算法
    [R,FM,P,MSE,K,SS] = Evaluation(result,Label,data);%对聚类的结果进行评价
    RR=[RR,R];
    FFM=[FFM,FM];
    PP=[PP,P];
    MMSE=[MMSE,MSE];
    KK=[KK,K];
    SSS=[SSS,SS];
end

disp(['R指标的平均值和方差分别为:',num2str(mean(RR)),'$\pm$',num2str(std(RR))]);
disp(['FM指标的平均值和方差分别为:',num2str(mean(FFM)),'$\pm$',num2str(std(FFM))]);
disp(['P指标的平均值和方差分别为:',num2str(mean(PP)),'$\pm$',num2str(std(PP))]);
disp(['MSE指标的平均值和方差分别为:',num2str(mean(MMSE)),'$\pm$',num2str(std(MMSE))]);
disp(['K指标的平均值和方差分别为:',num2str(mean(KK)),'$\pm$',num2str(std(KK))]);
disp(['CD指标的平均值和方差分别为:',num2str(mean(SSS)),'$\pm$',num2str(std(SSS))]);
if isempty(qresult)==1||num<k
    qresult=result;
end
Draw(mydata,qresult);
toc;

%主要是画图,画数据的散点图
function Draw( data,result )
%输入 data:输入数据    result:聚类结果的cell型 Label:数据的聚类结果,数组形式
Label=zeros(size(data,1),1);
for i=1:size(result,2)
    Label(result{i},1)=i;
end
for i=1:size(data,1)%扫描数据集
    if Label(i,1)==1
        plot(data(i,1),data(i,2),'.','Color',[0 0.545 0.271]);
        hold on;
    elseif Label(i,1)==2
        plot(data(i,1),data(i,2),'.','Color',[1 0 0]);
        hold on;
    elseif Label(i,1)==3
          plot(data(i,1),data(i,2),'.','Color',[1 0 1]);
        hold on;
    elseif Label(i,1)==4
        plot(data(i,1),data(i,2),'.','Color',[0,0,1]);
        hold on;
    elseif Label(i,1)==5
        plot(data(i,1),data(i,2),'.','Color',[1 0.40784 0.5451]);
        hold on;
    elseif Label(i,1)==6
       plot(data(i,1),data(i,2),'.','Color',[0.094 0.455 0.804]);
        hold on; 
    elseif Label(i,1)==7
        plot(data(i,1),data(i,2),'.','Color',[0 0.40784 0.5451]);
        hold on;
    end
end
end

