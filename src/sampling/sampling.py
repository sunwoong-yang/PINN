from pyDOE import *
# 얘는 기존 내 코드처럼 매 epoch마다 pde의 training points를 tensor에서 variable로 변환시킬 이유가 없어서임.
# 그리고 기존 내 코드는 bc, ic까지 variable로 변환시켰는데 이럴 이유는 없어보임 why: gradient 계산이 필요없이 그냥 network에 넣어서 output만 뽑아내면 되니까
def sampling(domain_bound,num_pts):
  samples = lhs(len(domain_bound), samples = num_pts, criterion = "maximin")
  for idx in range(len(domain_bound)):
    samples[:,idx] = samples[:,idx] * ( domain_bound[idx][1] - domain_bound[idx][0] ) + domain_bound[idx][0]
  return samples