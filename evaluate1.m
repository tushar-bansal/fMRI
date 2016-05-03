function err = evaluate1(outPut,YtestId)

for i = 1: size(outPut,1)
    
    y = YtestId(i,1);
    rank = find(outPut(i,:)==y);
    r(i)  = (rank-1)/59;
end
err= mean(r);