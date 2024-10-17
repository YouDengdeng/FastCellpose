function B=annotate(A)
A=imbinarize(A);
[m,n]=size(A);
B=zeros(m,n);
label=1; 
q=zeros(10000,3);
head=1;
tail=1;
adj_pos=[-1,-1;-1,0;-1,1;0,-1;0,1;1,-1;1,0;1,1]; 

for i=1:m
    for j=1:n
        if A(i,j)~=0&&B(i,j)==0
            B(i,j)=label;
            q(tail,:)=[i,j,1];  
            tail=tail+1;
            
            while head~=tail && q(head,3)==1  
                cur=q(head,:);
                for k=1:8
                    adj=cur(1:2)+adj_pos(k,:);
                    if adj(1)>=1&&adj(1)<=m&&adj(2)>=1&&adj(2)<=n   
                        if A(adj(1),adj(2))~=0&&B(adj(1),adj(2))==0 
                            B(adj(1),adj(2))=label;               
                            q(tail,:)=[adj(1),adj(2),1];           
                            tail=tail+1; 
                        end
                    end
                end
                head=head+1;       
            end
            area(label) = length(find(q~=0))/2;
            label=label+1;
            q=zeros(10000,3);
            head=1;
            tail=1;
        end
    end
end
B=uint8(B);