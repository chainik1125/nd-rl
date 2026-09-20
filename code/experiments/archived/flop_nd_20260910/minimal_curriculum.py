"""Released FLoP-style per-episode EMA advancement for the minimal-seed study."""
def update(cut,ema,episodes,reward,decay=.6,threshold=.9,force_after=20000):
    if not 0<=reward<=1 or not 0<=decay<1 or not 0<=threshold<=1:
        raise ValueError("Invalid curriculum probability setting")
    ema=decay*ema+(1-decay)*reward
    episodes+=1
    moved=cut>0 and (ema>threshold or (force_after>0 and episodes>force_after))
    return (cut-1,0.,0,True) if moved else (cut,ema,episodes,False)


if __name__=="__main__":
    state=(4,0.,0)
    for i in range(5):
        cut,ema,n,moved=update(*state,1)
        assert moved==(i==4)
        state=(cut,ema,n)
    assert state==(3,0.,0)
    assert update(4,.4,1,0)[:3]==(4,.24,2)
    assert update(4,0.,20000,0)==(3,0.,0,True)
    assert update(0,0.,20000,0)==(0,0.,20001,False)
    print("Four curriculum checks passed")
