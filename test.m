tic;
data=waveform;%��ȡ���ݼ�  ע��vehicle��magicҪ�ȴ���˳��
data =data(randperm(size(data,1)),:);
k=3;%���þ�����ص���Ŀ
col=size(data,2);%��ȡ���ݵ�����
mydata=data(:,1:col-1);%��ȡ���ݲ���
Label=data(:,col);%��ȡ���ݵ����ǩ
data=data(:,1:col-1);%��ȡ���ݲ���
data=zscore(data);%��׼���������ٵ�Ӱ��
N_MAX=7;%�ظ�ִ���㷨�Ĵ���
RR=[];%����������е�Rֵ
FFM=[];%����������е�FMֵ
PP=[];%����������е�Pֵ
MMSE=[];%����������е�NMIֵ
NNMI=[];
KK=[];
SSS=[];
qresult=[];
for i=1:N_MAX
%      result=SCIFWSA(data,k);%ִ��SCIFWSA�㷨
%      result=FSSCND(data,k);%ִ��FSSCND�㷨
%      result=DIFSC( data,k );%ִ��DIFSC�㷨
%      result= LRGR(data,k);%ִ��LRGR�㷨
    %result=MWND( data,k );%ִ��MWND�㷨
    [R,FM,P,MSE,K,SS] = Evaluation(result,Label,data);%�Ծ���Ľ����������
    RR=[RR,R];
    FFM=[FFM,FM];
    PP=[PP,P];
    MMSE=[MMSE,MSE];
    KK=[KK,K];
    SSS=[SSS,SS];
end

disp(['Rָ���ƽ��ֵ�ͷ���ֱ�Ϊ:',num2str(mean(RR)),'$\pm$',num2str(std(RR))]);
disp(['FMָ���ƽ��ֵ�ͷ���ֱ�Ϊ:',num2str(mean(FFM)),'$\pm$',num2str(std(FFM))]);
disp(['Pָ���ƽ��ֵ�ͷ���ֱ�Ϊ:',num2str(mean(PP)),'$\pm$',num2str(std(PP))]);
disp(['MSEָ���ƽ��ֵ�ͷ���ֱ�Ϊ:',num2str(mean(MMSE)),'$\pm$',num2str(std(MMSE))]);
disp(['Kָ���ƽ��ֵ�ͷ���ֱ�Ϊ:',num2str(mean(KK)),'$\pm$',num2str(std(KK))]);
disp(['CDָ���ƽ��ֵ�ͷ���ֱ�Ϊ:',num2str(mean(SSS)),'$\pm$',num2str(std(SSS))]);
if isempty(qresult)==1||num<k
    qresult=result;
end
Draw(mydata,qresult);
toc;

%��Ҫ�ǻ�ͼ,�����ݵ�ɢ��ͼ
function Draw( data,result )
%���� data:��������    result:��������cell�� Label:���ݵľ�����,������ʽ
Label=zeros(size(data,1),1);
for i=1:size(result,2)
    Label(result{i},1)=i;
end
for i=1:size(data,1)%ɨ�����ݼ�
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

