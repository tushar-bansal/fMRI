function acc = evaluate2(outPut,YtestId)


c=0;
for i = 1: size(outPut,1)
    
    y = YtestId(i,1); z = YtestId(i,2);

    r1 = find(outPut(i,:)==y); r2 = find(outPut(i,:)==z);
    if(r1<r2) c=c+1; end
end
acc = c/size(outPut,1);